# Modify the following paths
checkpoint_path = "./sdreamer/ckpt/BaseLine_Seq_ftALL_pl16_ns64_dm128_el2_dff512_eb0_bs8_f1_bs8/model_best.pth.tar"
output_path = "./output/"

# Do not change the following lines
dir_name = "baseline"
model_name = "BaseLine"
data_name = "Seq"
activation = "relu"
norm_type = "layernorm"
des_name = "bs8_NE"
patch_len = 16
seed = 42
ca_layers = 1
batch_size = 8
n_sequences = 64
ne_patch_len = 32
e_layers = 2
fold = 1

config = dict(
    # basic config
    seed=42,
    model="BaseLine",
    data=data_name,
    reload_ckpt=checkpoint_path,
    output_path=output_path,
    features="ALL",
    is_training=1,
    fold=fold,
    patch_len=patch_len,
    ne_patch_len=ne_patch_len,
    n_sequences=n_sequences,
    c_out=4,
    d_model=128,
    d_ff=512,
    e_layers=e_layers,
    ca_layers=ca_layers,
    dropout=0.1,
    path_drop=0.0,
    epochs=50,
    activation=activation,
    norm_type=norm_type,
    des=des_name,
    lr=0.001,
    weight_decay=0.0001,
    patience=30,
    batch_size=batch_size,
    output_attentions=True,
    model_id="test",
    useNorm=True,
    num_workers=10,
    seq_len=512,
    stride=16,
    pad=False,
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    # Formers
    mix_type=0,
    n_heads=8,
    seq_layers=2,
    pos_emb="learned",
    optimizer="adamw",
    beta_1=0.9,
    beta_2=0.999,
    eps=1e-9,
    scheduler="CosineLR",
    pct_start=0.3,
    step_size=30,
    gamma=0.5,
    use_gpu=False,
    gpu=1,
    use_multi_gpu=False,
    test_flop=False,
    print_freq=50,
)
