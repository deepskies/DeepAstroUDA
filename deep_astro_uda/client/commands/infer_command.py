from cleo.commands.command import Command

class InferCommand(Command):

    """
    Run inference pipeline for the client.
    """

    def handle(self):
        self.line('Running the client')
        self.call('client:infer')
