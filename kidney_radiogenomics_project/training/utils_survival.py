import torch

def cox_partial_likelihood_loss(logits, time, event):
    """
    logits: (B,) risk scores (higher => higher risk)
    time: (B,1) times
    event: (B,1) 1 if event observed, 0 if censored
    """
    # sort by time descending
    time = time.squeeze(1)
    event = event.squeeze(1)
    order = torch.argsort(time, descending=True)
    logits = logits[order]
    event = event[order]
    risk = logits.reshape(-1,1) - torch.logcumsumexp(logits, dim=0)
    # Only events contribute
    loss = - (risk.squeeze(1) * event).sum() / (event.sum() + 1e-6)
    return loss
