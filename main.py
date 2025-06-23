# Use LightningCLI to link arguments
class MyLightningCLI(L.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_classes", "model.init_args.num_classes", apply_on="instantiate")
        parser.link_arguments("data.class_weights", "model.init_args.class_weights", apply_on="instantiate")

# Running LightningCLI with linked arguments
if __name__ == "__main__":
    cli = MyLightningCLI(run=False)
    # Fit model
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # Run the test
    cli.trainer.test(ckpt_path='best', datamodule=cli.datamodule)
