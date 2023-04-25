exp_name = 'base'

G = 'CycleGenerator'
D = 'PatchDiscriminator'

data_train = dict(
    batch_size=16,
    data_dir='./data/LOL/Train/',
    ext='*.png',
    image_size=256,
    num_workers=8,
)

data_test = dict(
    batch_size=16,
    data_dir='./data/LOL/Eval/',
    ext='*.png',
    image_size=256,
    num_workers=8,
)

train = dict(
    n_epochs=1000,
    lr=1e-4,
    beta1=0.5,
    beta2=0.999,
    policy='color,translation,cutout',
    lambda_cycle=10,
    log_step=5,
    sample_every=10,
    checkpoint_every=50,
)