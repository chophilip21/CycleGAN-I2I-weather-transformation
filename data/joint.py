import numpy as np

class JointDataLoader(object):
    """
    Class for loading data from two dataloaders jointly,
    now with support for mid-epoch resume via state_dict().
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataloader_A = None
        self.dataloader_B = None
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = None
        self.is_train = None
        self.resume_iter = 0  # added for resume support

    def build(
        self, dataloader_A, dataloader_B, is_train, max_dataset_size=float("inf")
    ):
        self.dataloader_A = dataloader_A
        self.dataloader_B = dataloader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size
        self.is_train = is_train

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.dataloader_A_iter = iter(self.dataloader_A)
        self.dataloader_B_iter = iter(self.dataloader_B)
        self.iter = 0

        if self.is_train is False:
            np.random.seed(0)

        # Advance iterator if resuming mid-epoch
        if hasattr(self, "resume_iter") and self.resume_iter > 0:
            for _ in range(self.resume_iter):
                try:
                    next(self)
                except StopIteration:
                    break
            del self.resume_iter

        return self

    def __len__(self):
        if not self.is_train:
            return len(self.dataloader_A)
        return max(len(self.dataloader_A), len(self.dataloader_B))

    def __next__(self):
        A = None
        B = None
        try:
            A = next(self.dataloader_A_iter)
        except StopIteration:
            self.stop_A = True
            self.dataloader_A_iter = iter(self.dataloader_A)
            A = next(self.dataloader_A_iter)

        try:
            B = next(self.dataloader_B_iter)
        except StopIteration:
            self.stop_B = True
            self.dataloader_B_iter = iter(self.dataloader_B)
            B = next(self.dataloader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {
                "source": A,
                "target": B,
            }

    def state_dict(self):
        return {
            "iter": self.iter,
        }

    def load_state_dict(self, state):
        self.resume_iter = state.get("iter", 0)
