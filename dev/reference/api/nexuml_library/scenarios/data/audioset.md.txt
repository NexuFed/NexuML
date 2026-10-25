# nexuml_library.scenarios.data.audioset

*module*

AudioSet data specification builders.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/audioset.py#L1-L1)

## nexuml_library.scenarios.data.audioset.audioset_data

*function*

```python
def audioset_data(data_root: str = 'audioset_hf/full', train_subset: Literal['bal_train', 'unbal_train'] = 'bal_train', download: bool = False, sample_rate: int = 16000, clip_num_samples: int = 160000, batch_size: int = 128, num_workers: int = 4, num_classes: int = 527, perform_checks: bool = False, validate_layout: bool = True, max_samples: int | None = None) -> DataSpec
```

AudioSet balanced-train + eval data spec.

**Returns**

- `DataSpec` (`DataSpec`) — Audio dataset specification with fit and test splits.

**Raises**

- `FileNotFoundError` — If AudioSet data is missing or incomplete after
validation.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/data/audioset.py#L18-L89)
