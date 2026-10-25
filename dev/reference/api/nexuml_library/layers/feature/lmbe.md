# nexuml_library.layers.feature.lmbe

*module*

Log Mel Band Energy (LMBE) feature extractor layer.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L1-L1)

## nexuml_library.layers.feature.lmbe.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L13-L13)

## nexuml_library.layers.feature.lmbe.LMBE

*class*

```python
class LMBE(LayerDefinition)
```

Bases: `LayerDefinition`

Compute Log Mel Band Energies from a raw waveform.

Uses torchaudio's MelSpectrogram + AmplitudeToDB pipeline.
Optionally falls back to librosa for compatibility.

**Attributes**

- `n_mels` (`int`) — Number of Mel filterbanks.
- `n_fft` (`int`) — FFT size.
- `hop_length` (`int`) — Number of samples between frames.
- `win_length` (`int | None`) — Window length in samples.
- `power` (`int`) — Spectrogram power (1=magnitude, 2=power).
- `fmin` (`int`) — Minimum frequency.
- `fmax` (`int`) — Maximum frequency.
- `sr` (`int`) — Sample rate.
- `mel_scale` (`str`) — Mel scale type ("slaney" or "htk").
- `pad_mode` (`str`) — Padding mode for STFT.
- `use_librosa` (`bool`) — Use librosa backend instead of torchaudio.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L16-L52)

### nexuml_library.layers.feature.lmbe.LMBE.fmax

*attribute* · *class attribute* · *instance attribute*

```python
fmax: int = Field(default=8000, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L44-L44)

### nexuml_library.layers.feature.lmbe.LMBE.fmin

*attribute* · *class attribute* · *instance attribute*

```python
fmin: int = Field(default=0, ge=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L43-L43)

### nexuml_library.layers.feature.lmbe.LMBE.hop_length

*attribute* · *class attribute* · *instance attribute*

```python
hop_length: int = Field(default=512, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L40-L40)

### nexuml_library.layers.feature.lmbe.LMBE.mel_scale

*attribute* · *class attribute* · *instance attribute*

```python
mel_scale: str = 'slaney'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L45-L45)

### nexuml_library.layers.feature.lmbe.LMBE.n_fft

*attribute* · *class attribute* · *instance attribute*

```python
n_fft: int = Field(default=1024, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L39-L39)

### nexuml_library.layers.feature.lmbe.LMBE.n_mels

*attribute* · *class attribute* · *instance attribute*

```python
n_mels: int = Field(default=128, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L38-L38)

### nexuml_library.layers.feature.lmbe.LMBE.normalize

*attribute* · *class attribute* · *instance attribute*

```python
normalize: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L49-L49)

### nexuml_library.layers.feature.lmbe.LMBE.pad_mode

*attribute* · *class attribute* · *instance attribute*

```python
pad_mode: str = 'constant'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L46-L46)

### nexuml_library.layers.feature.lmbe.LMBE.power

*attribute* · *class attribute* · *instance attribute*

```python
power: int = Field(default=2, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L42-L42)

### nexuml_library.layers.feature.lmbe.LMBE.sample_rate

*attribute* · *class attribute* · *instance attribute*

```python
sample_rate: int = Field(default=16000, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L37-L37)

### nexuml_library.layers.feature.lmbe.LMBE.to_db

*attribute* · *class attribute* · *instance attribute*

```python
to_db: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L48-L48)

### nexuml_library.layers.feature.lmbe.LMBE.use_librosa

*attribute* · *class attribute* · *instance attribute*

```python
use_librosa: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L47-L47)

### nexuml_library.layers.feature.lmbe.LMBE.win_length

*attribute* · *class attribute* · *instance attribute*

```python
win_length: int | None = Field(default=None, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L41-L41)

### nexuml_library.layers.feature.lmbe.LMBE.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/feature/lmbe.py#L51-L52)
