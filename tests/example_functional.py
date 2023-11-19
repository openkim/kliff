import marimo

__generated_with = "0.1.36"
app = marimo.App()


@app.cell
def __():
    from kliff.models import KIMModel
    from kliff.dataset import Dataset
    return Dataset, KIMModel


@app.cell
def __(Dataset):
    dataset = Dataset("../tests/test_data/configs/Si_4.xyz", parser="ase", energy_key="Energy", forces_key="force")
    return dataset,


@app.cell
def __(dataset):
    configuration = dataset.get_configs()[1]
    return configuration,


@app.cell
def __(KIMModel):
    model = KIMModel("SW_StillingerWeber_1985_Si__MO_405512056662_006")
    return model,


@app.cell
def __(configuration, model):
    model(configuration)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
