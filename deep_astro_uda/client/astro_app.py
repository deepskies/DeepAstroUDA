from cleo.application import Application
#from commands.commands import DeepDanceRun

import sys

from deep_astro_uda.client.commands.demo_command import DemoCommand
from deep_astro_uda.client.commands.run_command import RunCommand
from deep_astro_uda.client.commands.infer_command import InferCommand

# TODO: Update using the correct commands.
# TODO: Add pydoc documentation.
dance_app = Application()
dance_app.add(DemoCommand())
dance_app.add(RunCommand())
dance_app.add(InferCommand())

def main() -> int:
    return dance_app.run()

if __name__ == '__main__':
    sys.exit(main())
