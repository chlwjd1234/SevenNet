import copy,os
from typing import Dict, List, Union
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import h5py
import numpy as np
from natsort import natsorted

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.train.graph_dataset import SevenNetGraphDataset

def _merge_feature_files(features_dir: Path) -> Dict:
    """Merge all batch feature files into a single file"""
    feature_files = natsorted(features_dir.glob("features_batch_*.pt"))
    all_features = []
    num_atoms = []
    
    for file in feature_files:
        saved_dict = torch.load(file)
        features = saved_dict["features"]
        batch_info = saved_dict["batch_info"]
        
        all_features.append(features)
        num_atoms.append(batch_info["num_atoms"].sum().item())

    # Convert to numpy arrays
    features_np = torch.cat(all_features, dim=0).numpy()
    num_atoms_np = np.array(num_atoms, dtype=np.int32)
    
    # Save to HDF5
    with h5py.File(features_dir / "all_features.h5", 'w') as f:
        # Create datasets with compression
        f.create_dataset('features', 
                        data=features_np,
                        chunks=True,
                        compression='gzip')
        f.create_dataset('num_atoms', 
                        data=num_atoms_np,
                        compression='gzip')
        
        # Add metadata
        f.attrs['total_structures'] = len(num_atoms)
        f.attrs['total_atoms'] = sum(num_atoms)

    merged_dict = {
        "features": torch.from_numpy(features_np),
        "num_atoms": num_atoms
    }

    return merged_dict

def extract_features(
    checkpoint: str,
    targets: Union[str, List[str]],
    output_dir: str,
    num_workers: int = 1,
    device: str = "cpu",
    batch_size: int = 4,
    **data_kwargs,
) -> None:
    """Extract node features from 7net model without force calculations"""
    # Setup feature extraction directory
    output_dir = Path(output_dir)
    features_dir = output_dir / "extracted_features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if not os.path.isfile(checkpoint):
        checkpoint = util.pretrained_name_to_path(checkpoint)
    model, _ = util.model_from_checkpoint(checkpoint)
    cutoff = model.cutoff

    # Create modified model copy without gradient modules
    model_copy = copy.deepcopy(model)
    removed_modules = {}
    
    # Remove force output and related gradient modules
    
    modules_to_remove = ['force_output', 'reduce_total_enegy']
    for name in modules_to_remove:
        if name in model_copy._modules:
            removed_modules[name] = model_copy._modules[name]
            del model_copy._modules[name]
    
    # Handle targets
    if isinstance(targets, str):
        targets = [targets]

    # Create dataset
    dataset = []
    for file in targets:
        tmplist = SevenNetGraphDataset.file_to_graph_list(
            filename=file,
            cutoff=cutoff,
            num_cores=num_workers,
            **data_kwargs,
        )
        dataset.extend(tmplist)

    if len(dataset) == 0:
        raise RuntimeError("No valid data found in targets")

    # Setup data loader and model
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model_copy.to(device)
    model_copy.set_is_batch_data(True)
    model_copy.eval()

    print(f"Extracting features from {len(dataset)} structures...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            output = model_copy(batch)
    
            # Extract features
            features = output[KEY.NODE_FEATURE]
            atomic_energies = output.get(KEY.SCALED_ATOMIC_ENERGY, None)
            
            batch_info = {
                #"atomic_numbers": output[KEY.ATOMIC_NUMBERS],
                "num_atoms": output[KEY.NUM_ATOMS]
            }
            
            save_dict = {
                "features": features.cpu(),
                "batch_info": {k: v.cpu() for k, v in batch_info.items()}
            }
            
            torch.save(save_dict, features_dir / f"features_batch_{batch_idx}.pt")

    print("Merging feature files...")
    _merge_feature_files(features_dir)

        # Clean up temporary PT files
    for file in features_dir.glob("features_batch_*.pt"):
        file.unlink()
        
    print(f"Features saved to {features_dir}/all_features.h5")
