class RandomGaussianBlurConfig:
    def __init__(self, p=0.5, max_gaussian_kernel=19) -> None:
        self.p = p
        self.max_gaussian_kernel = max_gaussian_kernel


class DataAugmentationConfig:
    def __init__(self) -> None:
        self.mean_normalization = [0.5, 0.5, 0.5]
        self.std_normalization = [0.5, 0.5, 0.5]
        self.image_gaussian_config = RandomGaussianBlurConfig(
            p=0.5,
            max_gaussian_kernel=19,
        )
        self.depth_gaussian_config = RandomGaussianBlurConfig(
            p=0.5,
            max_gaussian_kernel=36,
        )
        self.random_horizontal_flip_prob = 0.5
