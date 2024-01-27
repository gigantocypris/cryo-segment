'''
# The following iterates over all datasets in the portal, then all runs per dataset, then all tomograms per run

from cryoet_data_portal import Client, Dataset

# Instantiate a client, using the data portal GraphQL API by default
client = Client()

# Iterate over all datasets
for dataset in Dataset.find(client):
    print(f"Dataset: {dataset.title}")
    for run in dataset.runs:
        print(f"  - run: {run.name}")
        for tvs in run.tomogram_voxel_spacings:
            print(f"    - voxel spacing: {tvs.voxel_spacing}")
            for tomo in tvs.tomograms:
                print(f"        - tomo: {tomo.name}")
'''

# The following iterates over all tomograms related to a specific organism and downloads a 25% scale preview tomogram in MRC format for each one.
                
import json

from cryoet_data_portal import Client, Tomogram

# Instantiate a client, using the data portal GraphQL API by default
client = Client()

# Find all tomograms related to a specific organism
tomos = Tomogram.find(
    client,
    [
        Tomogram.tomogram_voxel_spacing.run.dataset.organism_name
        == "Schizosaccharomyces pombe"
    ],
)
for tomo in tomos:
    # Access any useful metadata for each tomogram
    print(tomo.name)

    # Print the tomogram metadata as a json string
    print(json.dumps(tomo.to_dict(), indent=4))

    # Download a tomogram in the MRC format (uncomment to actually download files)
    tomo.download_mrcfile()