import intake

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# Look up a driver class by its name in the registry.
catalog_class = intake.registry["bluesky-mongo-normalized-catalog"]

bluesky_cartpole_catalog_instance = catalog_class(
    metadatastore_db="mongodb://localhost:27017/md",
    asset_registry_db="mongodb://localhost:27017/ar",
)
