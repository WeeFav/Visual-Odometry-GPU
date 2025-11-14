import numpy as np
import matplotlib.pyplot as plt

def compute_ate(gt, est):
    errors = np.linalg.norm(gt - est, axis=1)
    return np.sqrt(np.mean(errors**2)), errors

def compute_rpe(gt, est, delta=1):
    rpe = []
    for i in range(len(gt) - delta):
        gt_rel = gt[i+delta] - gt[i]
        est_rel = est[i+delta] - est[i]
        rpe.append(np.linalg.norm(gt_rel - est_rel))
    rpe = np.array(rpe)

    return np.sqrt(np.mean(rpe**2)), rpe

def compute_scale_drift(scale):
    scale_ratio = scale[:, 1] / scale[:, 0]
    return np.mean(scale_ratio), scale_ratio

def kitti_drift(gt, est, segment_lengths=[100]):
    results = {}
    dist = np.cumsum(np.linalg.norm(gt[1:] - gt[:-1], axis=1))
    
    for L in segment_lengths:
        drift_list = []
        for i in range(len(dist)):
            # find end index
            end = np.searchsorted(dist, dist[i] + L)
            if end >= len(gt):
                break

            gt_disp = gt[end] - gt[i]
            est_disp = est[end] - est[i]
            
            trans_error = np.linalg.norm(gt_disp - est_disp)
            drift_percent = 100.0 * trans_error / L

            drift_list.append(drift_percent)
        
        results[L] = np.mean(drift_list)

    return results

if __name__ == '__main__':
    gt = np.loadtxt("./build/gt_path.txt")        # shape Nx2
    est = np.loadtxt("./build/est_path.txt")      # shape Nx2
    scale = np.loadtxt("./build/scale.txt")      # shape Nx2
    assert gt.shape == est.shape

    ate, ates = compute_ate(gt, est)

    rpe, rpes = compute_rpe(gt, est, delta=1)

    avg_scale_drift, scale_drift = compute_scale_drift(scale)

    kitti = kitti_drift(gt, est, segment_lengths=[50, 100, 200])
    print("KITTI-style drift:", kitti)

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(np.arange(len(gt)), ates)
    axes[0, 0].set_title(f"ATE RMSE: {ate:.2f}")
    
    axes[0, 1].plot(np.arange(1, len(gt)), rpes)
    axes[0, 1].set_title(f"RPE RMSE: {rpe:.2f}")

    axes[1, 0].plot(np.arange(1, len(gt)), scale_drift)
    axes[1, 0].set_title(f"Scale Drift Avg: {avg_scale_drift:.2f}")
    
    fig.suptitle("Metrics")
    plt.tight_layout()
    plt.show()