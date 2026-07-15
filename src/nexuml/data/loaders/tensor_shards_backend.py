class TensorShardsLoaderBackend:
    def create_loader(
        self,
        module,
        dataset,
        *,
        split: str,
        shuffle: bool = False,
        sampler=None,
    ):
        params = module.loader_spec.params

        return TensorShardWindowLoader(
            dataset=dataset,
            split=split,
            batch_size=module.loader_spec.batch_size,
            shards_per_window=int(params.get("shards_per_window", 6)),
            prefetch_windows=int(params.get("prefetch_windows", 2)),
            prefetch_workers=int(params.get("prefetch_workers", 2)),
            shuffle_shards=bool(params.get("shuffle_shards", shuffle)),
            shuffle_samples=bool(params.get("shuffle_samples", shuffle)),
            pin_memory=bool(params.get("pin_memory", False)),
        )
