from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from utils import seconds_to_frames

class Pooling:
    
    def __init__(self, feature_dir, boundary_dir, pca_components=350):
        self.feature_dir = Path(feature_dir)
        self.boundary_dir = Path(boundary_dir)
        self.scaler_model = None
        self.pca_model = None
        self.pca_components = pca_components
        self.output_dir = self.set_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_path = self.output_dir / "scaler_model.npy"
        self.pca_path = self.output_dir / "pca_model.npy"

    def set_output_dir(self):

        output_dir = "/".join(self.feature_dir.parts[-4:])
        output_dir = Path("pooled_features") / output_dir / "/".join(self.boundary_dir.parts[-2:]) 
        print(f"Setting output directory to: {output_dir}")
        return output_dir
    
    def fit(self):
        if self.scaler_path.exists() and self.pca_path.exists():
            print("Loading existing scaler and PCA models...")
            self.scaler_model = np.load(self.scaler_path, allow_pickle=True).item()
            self.pca_model = np.load(self.pca_path, allow_pickle=True).item()
            return

        feature_files = sorted(self.feature_dir.rglob("*.npy"))
        if len(feature_files) > 10_000:
            print(f"Warning: Found {len(feature_files)} feature files. Using fit_batched() to avoid memory issues.")
            self.fit_batched()
            return

        print("Fitting scaler and PCA on all features at once...")

        features = [np.load(f) for f in tqdm(feature_files, desc="Loading features")]
        features = np.vstack(features)

        if not self.scaler_path.exists():
            self.scaler_model = StandardScaler()
            self.scaler_model.fit(features)
            np.save(self.scaler_path, self.scaler_model)
        else:
            self.scaler_model = np.load(self.scaler_path, allow_pickle=True).item()

        scaled_features = self.scaler_model.transform(features)

        if not self.pca_path.exists():
            self.pca_model = PCA(n_components=self.pca_components)
            self.pca_model.fit(scaled_features)
            np.save(self.pca_path, self.pca_model)
        else:
            self.pca_model = np.load(self.pca_path, allow_pickle=True).item()

    def fit_batched(self, batch_size=10_000):
        
        if self.scaler_path.exists() and self.pca_path.exists():
            print("Loading existing scaler and PCA models...")
            self.scaler_model = np.load(self.scaler_path, allow_pickle=True).item()
            self.pca_model = np.load(self.pca_path, allow_pickle=True).item()
            return
        
        feature_files = sorted(self.feature_dir.rglob("*.npy"))
        print(f"Fitting scaler in batches of {batch_size}...")
        self.scaler_model = StandardScaler()

        for batch_start in tqdm(range(0, len(feature_files), batch_size), desc="Fitting scaler"):
            batch_files = feature_files[batch_start:batch_start + batch_size]
            batch_features = [np.load(f) for f in batch_files]
            batch_features = np.vstack(batch_features)
            self.scaler_model.partial_fit(batch_features)
        np.save(self.scaler_path, self.scaler_model)
        
        self.pca_model = IncrementalPCA(n_components=self.pca_components)
        print(f"Fitting PCA in batches of {batch_size}...")
        for batch_start in tqdm(range(0, len(feature_files), batch_size), desc="Fitting PCA"):
            batch_files = feature_files[batch_start:batch_start + batch_size]
            batch_features = [np.load(f) for f in batch_files]
            batch_features = np.vstack(batch_features)
            batch_features_scaled = self.scaler_model.transform(batch_features)
            self.pca_model.partial_fit(batch_features_scaled)
        np.save(self.pca_path, self.pca_model)
    
    def transform(self):

        feature_files = sorted(self.feature_dir.rglob("*.npy"))

        all_boundaries = {}
        for boundary_file in sorted(self.boundary_dir.rglob("*.list")):
            all_boundaries[boundary_file.stem] = np.loadtxt(boundary_file)
        
        for feature_file in tqdm(feature_files, desc="Transforming features"):
            features = np.load(feature_file)
            if features.size == 0: continue
            if features.ndim == 1:
                features = features.reshape(1, -1)
                
            scaled_features = self.scaler_model.transform(features)
            pca_features = self.pca_model.transform(scaled_features)
        
            segment_name = feature_file.stem
            if segment_name in all_boundaries:
                boundaries = all_boundaries[segment_name]
                if not isinstance(boundaries, np.ndarray):
                    boundaries = np.array(boundaries)
                if boundaries.ndim == 0:
                    boundaries = np.array([boundaries])

                boundaries = [seconds_to_frames(b) for b in boundaries]
                start = 0
                for end in boundaries:
                    out_path = self.output_dir / feature_file.relative_to(self.feature_dir).parent / f"{feature_file.stem}_{start}_{end}.npy"
                    
                    if out_path.exists():
                        start = end
                        continue

                    if end == start: continue

                    segment = pca_features[start:end]
                    if segment.size == 0: continue

                    pooled_segment = np.mean(segment, axis=0)
                    pooled_feature = np.asarray(pooled_segment, dtype=np.float32)       

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, pooled_feature)
                    start = end

def load_pooled_features(feature_dir):

    feature_files = sorted(feature_dir.rglob("*.npy"))
    features = []
    filenames = []
    intervals = []
    for feature_file in tqdm(feature_files, desc="Loading pooled features"):
        if "model" in feature_file.stem: continue

        pooled_feature = np.load(feature_file)
        features.append(pooled_feature)

        start_frame, end_frame = map(int, feature_file.stem.split("_")[-2:])
        filenames.append(feature_file.stem.rsplit("_", 2)[0])
        intervals.append((start_frame, end_frame))

    features = np.asarray(features, dtype=np.float32)
    # features = normalize(features, axis=1, norm="l2").astype(np.float32)
    features = features - np.mean(features, axis=0)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

    print(
        f"Loaded {len(features)} pooled feature files. "
        f"Ex: {features[0].shape} from {filenames[0]} with interval {intervals[0]}"
    )
    return features, filenames, intervals

def main(args):
    pooling = Pooling(args.feature_dir, args.boundary_dir, pca_components=args.n_components)
    pooling.fit()
    pooling.transform()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pool features based on boundaries")
    parser.add_argument("feature_dir", type=Path, help="Directory from which to extract audio.")
    parser.add_argument("boundary_dir", type=Path, help="Directory containing boundary information.")
    parser.add_argument("--n_components", type=int, default=350, help="Number of PCA components.")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Batch size for fitting scaler and PCA")
    args = parser.parse_args()
    main(args)