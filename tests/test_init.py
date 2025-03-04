def test_package_import() -> None:
    """Test that the bada package can be imported."""
    import bada

    assert hasattr(bada, "__version__")
    assert isinstance(bada.__version__, str)

    assert hasattr(bada, "models")
    assert hasattr(bada, "parsers")
    assert hasattr(bada, "processing")


def test_models_import() -> None:
    """Test that the models module can be imported."""
    from bada import models

    assert hasattr(models, "DSFInput")
    assert hasattr(models, "LightCycler480Raw")
    assert hasattr(models, "QuantStudio7Raw")
