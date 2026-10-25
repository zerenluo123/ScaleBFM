import numpy as np
from scipy.signal import butter, filtfilt


def rate_limit(x, max_step):
    """Clamp per-sample change to max_step; forward and backward passes are averaged so the result
    does not lag the input, and the average still respects the limit."""

    def forward(z):
        y = z.copy()
        for i in range(1, len(y)):
            y[i] = np.clip(z[i], y[i - 1] - max_step, y[i - 1] + max_step)
        return y

    return 0.5 * (forward(x) + forward(x[::-1])[::-1])


def smooth_root_height(z, fps, cutoff_hz, max_speed):
    """Zero-phase low-pass followed by a speed limit, for a whole clip of root heights."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) > 9:
        b, a = butter(2, cutoff_hz / (fps / 2))
        z = filtfilt(b, a, z, method="gust")
    return rate_limit(z, max_speed / fps)


def filter_root_height(frames, root_name, fps, cutoff_hz, max_speed, min_height=None):
    """Shift every bone of each frame by the same vertical offset so the root height follows the
    smoothed curve; the pose and all orientations are unchanged. With min_height, frames whose root is
    lower (kneeling, seated) are first raised to it, so the pose above the root is kept and the legs
    are left below the floor."""
    z = np.array([frame[root_name][0][2] for frame in frames])
    target = z if min_height is None else np.maximum(z, min_height)
    dz = smooth_root_height(target, fps, cutoff_hz, max_speed) - z
    shifted = []
    for frame, d in zip(frames, dz):
        offset = np.array([0.0, 0.0, d])
        shifted.append({bone: (np.asarray(pos) + offset, quat) for bone, (pos, quat) in frame.items()})
    return shifted
