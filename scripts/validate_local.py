# Standard library
import ast
import sys
from pathlib import Path
# Utils
from src.utils.vertex import render_job_spec

# Static, TensorFlow-free pre-flight checks — catches the class of bug that doesn't need
# GPU, Docker, or Vertex AI to reproduce (an argparse choices list out of sync with its
# callers, a rendered command list missing a flag Dockerfile expects). Deliberately never
# imports src/training/train.py or api/main.py directly: both pull in tensorflow at
# module level, which is exactly the cost this script exists to avoid paying. train.py's
# argparse config is instead read via ast, off the source text, without executing it.

ROOT = Path( __file__ ).resolve().parent.parent


def extract_stage_choices( train_py_path: Path ) -> set[ str ]:
	'''Read --stage's choices=[...] out of train.py's argparse setup via ast, without importing the module (and therefore without importing tensorflow).'''

	tree = ast.parse( train_py_path.read_text(), filename=str( train_py_path ) )

	for node in ast.walk( tree ):
		if not isinstance( node, ast.Call ):
			continue

		if not ( isinstance( node.func, ast.Attribute ) and node.func.attr == 'add_argument' ):
			continue

		if not ( node.args and isinstance( node.args[ 0 ], ast.Constant ) and node.args[ 0 ].value == '--stage' ):
			continue

		for kw in node.keywords:
			if kw.arg == 'choices' and isinstance( kw.value, ast.List ):
				return { elt.value for elt in kw.value.elts if isinstance( elt, ast.Constant ) }

	raise AssertionError( f'No --stage argument with choices=[...] found in { train_py_path }' )


def extract_submitted_stages( search_root: Path ) -> dict[ str, list[ str ] ]:
	'''Ast-scan every .py file under search_root for submit_vertex_job(..., "literal") calls, returning {file: [stage_values]} for every literal-string stage argument found.'''

	found: dict[ str, list[ str ] ] = {}

	for py_file in search_root.rglob( '*.py' ):
		if '.venv' in py_file.parts:
			continue

		tree = ast.parse( py_file.read_text(), filename=str( py_file ) )

		for node in ast.walk( tree ):
			if not ( isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id == 'submit_vertex_job' ):
				continue

			if len( node.args ) >= 2 and isinstance( node.args[ 1 ], ast.Constant ) and isinstance( node.args[ 1 ].value, str ):
				found.setdefault( str( py_file.relative_to( ROOT ) ), [] ).append( node.args[ 1 ].value )

	return found


def extract_dockerfile_uv_sync_flags( dockerfile_path: Path ) -> set[ str ]:
	'''
	Pull the flag list off Dockerfile's `uv sync` command line — the build-time
	counterpart to vertex_job.yaml's runtime `uv run` command. Scans line-by-line and only
	matches a line whose stripped content starts with `uv sync` (i.e. the actual
	instruction), not a substring match against the whole file — this Dockerfile's own
	comments mention "the uv sync layer" in prose, which a plain regex.search would match
	first and silently return the wrong line's flags.
	'''

	for line in dockerfile_path.read_text().splitlines():
		stripped = line.strip()

		if stripped.startswith( 'uv sync' ):
			return set( stripped.removeprefix( 'uv sync' ).split() )

	raise AssertionError( f'No `uv sync` command line found in { dockerfile_path }' )


def main() -> None:

	failures: list[ str ] = []

	# Check 1 — every literal stage string passed to submit_vertex_job() must be one of
	# train.py's declared --stage choices. This is the exact bug class the last handoff
	# brief caught after a real job failure: fix it here, for free, before that.
	stage_choices = extract_stage_choices( ROOT / 'src' / 'training' / 'train.py' )
	submitted_stages = extract_submitted_stages( ROOT )

	print( f'--stage choices (train.py): { sorted( stage_choices ) }' )

	for file_path, stages in submitted_stages.items():
		for stage in stages:
			print( f'  { file_path } calls submit_vertex_job( ..., \'{ stage }\' )' )

			if stage not in stage_choices:
				failures.append(
					f'{ file_path } passes stage=\'{ stage }\', not in train.py --stage choices { sorted( stage_choices ) }'
				)

	# Check 2 — render infra/vertex_job.yaml for both real stage values and print the
	# exact command/args Vertex AI would receive, with no API call made.
	for stage in sorted( stage_choices ):
		display_name, worker_pool_specs = render_job_spec( 'validate-local', stage )
		container_spec = worker_pool_specs[ 0 ][ 'container_spec' ]

		print( f'\nRendered spec for stage=\'{ stage }\':' )
		print( f'  command: { container_spec[ "command" ] }' )
		print( f'  args:    { container_spec[ "args" ] }' )

		if '--stage' not in container_spec[ 'args' ]:
			failures.append( f'stage=\'{ stage }\': rendered args missing --stage flag entirely: { container_spec[ "args" ] }' )
			continue

		stage_flag_index = container_spec[ 'args' ].index( '--stage' )

		if container_spec[ 'args' ][ stage_flag_index + 1 ] != stage:
			failures.append(
				f'stage=\'{ stage }\': rendered --stage value is \'{ container_spec[ "args" ][ stage_flag_index + 1 ] }\', expected \'{ stage }\''
			)

	# Check 3 — vertex_job.yaml's `uv run` flags must match Dockerfile's `uv sync` flags.
	# This is the other bug class from the same handoff brief: a fix landing in one file
	# (--frozen --no-dev added to the YAML) without the equivalent flags existing on the
	# Dockerfile side, or vice versa, silently forcing a full dependency resync at runtime.
	_, base_worker_pool_specs = render_job_spec( 'validate-local', sorted( stage_choices )[ 0 ] )
	command = base_worker_pool_specs[ 0 ][ 'container_spec' ][ 'command' ]

	if 'run' not in command or 'python' not in command:
		failures.append( f'vertex_job.yaml command list missing \'run\' or \'python\': { command }' )
	else:
		vertex_flags = set( command[ command.index( 'run' ) + 1 : command.index( 'python' ) ] )
		docker_flags = extract_dockerfile_uv_sync_flags( ROOT / 'Dockerfile' )

		print( f'\nvertex_job.yaml `uv run` flags: { sorted( vertex_flags ) }' )
		print( f'Dockerfile `uv sync` flags:     { sorted( docker_flags ) }' )

		if vertex_flags != docker_flags:
			failures.append(
				f'vertex_job.yaml `uv run` flags { sorted( vertex_flags ) } != Dockerfile `uv sync` flags { sorted( docker_flags ) }'
			)

	# Summary
	print()

	if failures:
		print( f'FAILED — { len( failures ) } issue(s):' )

		for failure in failures:
			print( f'  - { failure }' )

		sys.exit( 1 )

	print( 'PASSED — all local pre-flight checks OK' )


if __name__ == '__main__':
	main()
