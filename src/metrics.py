import numpy as np

WET_THRESHOLD = 0.1


def field_metrics(pred, true, thr=WET_THRESHOLD, depth_domain="both"):
    pred = np.asarray(pred)
    true = np.asarray(true)
    assert pred.shape == true.shape, f"{pred.shape} vs {true.shape}"

    pw = pred > thr
    tw = true > thr
    TP = int((pw & tw).sum())
    FP = int((pw & ~tw).sum())
    FN = int((~pw & tw).sum())
    TN = int((~pw & ~tw).sum())

    if depth_domain == "both":
        dom = pw & tw
    elif depth_domain == "union":
        dom = pw | tw
    else:
        raise ValueError(depth_domain)

    if dom.any():
        d = pred[dom].astype(np.float64) - true[dom].astype(np.float64)
        rmse = float(np.sqrt((d ** 2).mean()))
        mae = float(np.abs(d).mean())
        bias = float(d.mean())
    else:
        rmse = mae = bias = float("nan")

    return dict(
        CSI=TP / max(TP + FP + FN, 1),
        FAR=FP / max(TP + FP, 1),
        POD=TP / max(TP + FN, 1),
        RMSE=rmse, MAE=mae, Bias=bias,
        TP=TP, FP=FP, FN=FN, TN=TN,
    )


def peak_timing_error(pred, true, thr=WET_THRESHOLD, dt=1.0):
    a_p = (pred > thr).sum((1, 2))
    a_t = (true > thr).sum((1, 2))
    return float((int(np.argmax(a_p)) - int(np.argmax(a_t))) * dt / 3600.0)


def metrics_per_timestep(pred, true, thr=WET_THRESHOLD):
    return [dict(t=t, **field_metrics(pred[t], true[t], thr))
            for t in range(pred.shape[0])]


class MetricAccumulator:

    def __init__(self, thr=WET_THRESHOLD, depth_domain="both"):
        self.thr = thr
        self.depth_domain = depth_domain
        self.TP = self.FP = self.FN = self.TN = 0
        self.sq = self.abs = self.sum = 0.0
        self.n = 0
        self.area_pred = []
        self.area_true = []

    def update(self, pred, true):
        pw = pred > self.thr
        tw = true > self.thr
        self.TP += int((pw & tw).sum())
        self.FP += int((pw & ~tw).sum())
        self.FN += int((~pw & tw).sum())
        self.TN += int((~pw & ~tw).sum())

        dom = (pw & tw) if self.depth_domain == "both" else (pw | tw)
        if dom.any():
            d = pred[dom].astype(np.float64) - true[dom].astype(np.float64)
            self.sq += float((d ** 2).sum())
            self.abs += float(np.abs(d).sum())
            self.sum += float(d.sum())
            self.n += int(dom.sum())
        self.area_pred.append(int(pw.sum()))
        self.area_true.append(int(tw.sum()))

    def result(self):
        TP, FP, FN = self.TP, self.FP, self.FN
        out = dict(
            CSI=TP / max(TP + FP + FN, 1),
            FAR=FP / max(TP + FP, 1),
            POD=TP / max(TP + FN, 1),
            TP=TP, FP=FP, FN=FN, TN=self.TN,
        )
        if self.n > 0:
            out["RMSE"] = float(np.sqrt(self.sq / self.n))
            out["MAE"] = float(self.abs / self.n)
            out["Bias"] = float(self.sum / self.n)
        else:
            out["RMSE"] = out["MAE"] = out["Bias"] = float("nan")
        if self.area_pred:
            out["peak_timestep_error"] = int(
                np.argmax(self.area_pred) - np.argmax(self.area_true))
        return out
