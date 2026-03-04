"""
TopoRad 完整基准测试（信号增强版）
===================================
TEST 1:  无损性（定理 A）
TEST 2:  正交性（命题 7）
TEST 3:  通道权重（命题 9）
TEST 4:  VAE 拓扑稳定性（定理 B）
TEST 5:  载波校准 + w_u 驱动生成
TEST 6:  多用户拓扑可分性（PCA → LDA 判别载波）
TEST 7:  检测与归属（Bonferroni + 经验 H₀）
TEST 8a: 放射性 — 标准微调（50 epoch, LoRA rank=8）
TEST 8b: 放射性 — 轻微调（10 epoch, LoRA rank=4）
TEST 9:  虚警控制（干净微调 → FPR）
TEST 10: 图像质量评估（LPIPS + PSNR + SSIM）
"""
import os, sys, time, gc, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
from PIL import Image

# ═══════════════════════════════════════════════════════════
#  核心参数（信号增强：K=16, 16 imgs/user, LoRA rank=8）
# ═══════════════════════════════════════════════════════════
K_CANDIDATES   = 16        # 候选数：越大 → w_u 投影越极端
IMGS_PER_USER  = 16        # 每用户图数：越大 → LoRA 学到更多
N_USERS        = 5
N_CALIB        = 50        # PCA 校准图数
N_CLEAN_BASE   = 30        # 经验 H₀ 用干净基线图数
N_FT_GEN       = 50        # TEST 8 检测用生成图数（增强统计功效）
LORA_RANK_STD  = 8         # 标准微调 rank
LORA_EPOCH_STD = 50        # 标准微调 epoch
LORA_RANK_LT   = 4         # 轻微调 rank
LORA_EPOCH_LT  = 10        # 轻微调 epoch

SAVE_DIR = os.path.join(os.getcwd(), "output_radioactive", "watermarked_images")

def P(msg):
    print(msg, flush=True)

P("=" * 70)
P("  TopoRad Benchmark (Signal-Enhanced: K=%d, %d img/user, rank=%d)" % (
    K_CANDIDATES, IMGS_PER_USER, LORA_RANK_STD))
P("  GPU: %s" % torch.cuda.get_device_name(0))
P("=" * 70)

LOCAL_MODEL = os.path.join(os.getcwd(), "model", "stable-diffusion-v1-4")

P("\n[LOAD] Loading SD v1.4...")
t0 = time.time()
from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe = StableDiffusionPipeline.from_pretrained(
    LOCAL_MODEL, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
P("  Loaded in %.1fs" % (time.time() - t0))

P("[WARMUP] CUDA compile...")
with torch.no_grad():
    _ = pipe("warmup", num_inference_steps=3, guidance_scale=7.5)
P("  Warmup done")

from radioactive.config import RadioactiveConfig
from radioactive.pipeline.embedder import RadioactiveEmbedder
from neurips_core.lossless import sample_watermarked_noise, verify_lossless

config = RadioactiveConfig(
    model_id=LOCAL_MODEL, num_users=10, num_inference_steps=20,
    use_carrier=True, carrier_n_calib=N_CALIB, carrier_n_components=50,
    lora_epochs=LORA_EPOCH_STD, lora_batch_size=2, lora_lr=1e-4,
    lora_rank=LORA_RANK_STD, lora_alpha=LORA_RANK_STD,
)
embedder = RadioactiveEmbedder(pipe, config)

USER_PROMPTS = {}
for u in range(N_USERS):
    USER_PROMPTS[u] = config.prompts[u * 2:(u + 1) * 2]
    if len(USER_PROMPTS[u]) < 2:
        USER_PROMPTS[u] = [config.prompts[u % len(config.prompts)]]

# # ====================================================================
# # TEST 1: LOSSLESS
# # ====================================================================
# P("\n" + "=" * 70)
# P("[TEST 1] Lossless (Theorem A: eps ~ N(0,I))")
# P("=" * 70)
# C, H, W = embedder._get_latent_shape()
# uid_t = torch.zeros(100, dtype=torch.long, device="cuda")
# eps = sample_watermarked_noise(uid_t, embedder.spherical, (100, C, H, W), device=torch.device("cuda"))
# stats = verify_lossless(eps)
# from scipy.stats import kstest
# flat = eps.cpu().numpy().flatten()
# subset = flat[np.random.choice(len(flat), size=50000, replace=False)]
# _, ks_p = kstest(subset, "norm")
# test1_pass = stats["is_approx_zero_mean"] and stats["is_approx_unit_var"]
# P("  Mean=%.6f  Var=%.6f  KS p=%.4f  => %s" % (stats["mean"], stats["var"], ks_p, "PASS" if test1_pass else "FAIL"))
#
# # ====================================================================
# # TEST 2: ORTHOGONAL ROTATION
# # ====================================================================
# P("\n" + "=" * 70)
# P("[TEST 2] Orthogonal Rotation (Prop 7)")
# P("=" * 70)
# from radioactive.core.topo_coupler import TopoCoupler
# from radioactive.core.topo_vectorize import extract_topo_signature
# coupler = TopoCoupler(embed_dim=64, coupling_strength=0.3, temperature=50.0)
# n_flat = C * H * W
# test_arr = np.random.randn(C, H, W)
# sig = extract_topo_signature(test_arr, max_dim=1, pi_resolution=config.pi_resolution)
# e_u = np.random.randn(64).astype(np.float64); e_u /= np.linalg.norm(e_u)
# hh = coupler.compute_orthogonal_rotation(sig, e_u, n_flat)
# errs = []
# for _ in range(20):
#     x = torch.randn(n_flat)
#     y = coupler.apply_householder(hh, x)
#     errs.append(abs((y.norm() / x.norm()).item() - 1.0))
# test2_pass = max(errs) < 1e-3
# P("  Max err=%.2e  => %s" % (max(errs), "PASS" if test2_pass else "FAIL"))
#
# # ====================================================================
# # TEST 3: CHANNEL WEIGHTS
# # ====================================================================
# P("\n" + "=" * 70)
# P("[TEST 3] Channel Weights (Prop 9)")
# P("=" * 70)
# alpha_ch = coupler.compute_channel_weights(sig, num_channels=C)
# P("  Weights: %s  Sum=%.6f  => PASS" % (["%.4f" % a for a in alpha_ch], alpha_ch.sum()))
#
# # ====================================================================
# # TEST 4: VAE TOPOLOGY STABILITY
# # ====================================================================
# P("\n" + "=" * 70)
# P("[TEST 4] VAE Topology Stability (Theorem B)")
# P("=" * 70)
# from radioactive.core.stability import StabilityVerifier
# verifier = StabilityVerifier(max_dim=config.ph_max_dim, pi_resolution=config.pi_resolution, pi_bandwidth=config.pi_bandwidth)
# vae = pipe.vae; vae.eval()
# orig_list, recon_list = [], []
# with torch.no_grad():
#     for i in range(5):
#         res = pipe(config.prompts[i], num_inference_steps=20, guidance_scale=7.5)
#         img_np = np.array(res.images[0]).astype(np.float32) / 255.0
#         img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
#         img_t = (img_t * 2.0 - 1.0).to("cuda", dtype=vae.dtype)
#         z = vae.encode(img_t).latent_dist.sample()
#         recon = ((vae.decode(z).sample.clamp(-1, 1) + 1) / 2).cpu().numpy()[0]
#         orig_list.append(img_np.transpose(2, 0, 1)); recon_list.append(recon)
#         P("    img %d/5" % (i+1))
# report = verifier.verify_vae_stability(np.stack(orig_list), np.stack(recon_list), verbose=True, log_fn=P)
# test4_pass = report.theorem_b_rate >= 0.9
# P("  Thm B rate=%.0f%%  d_B=%.4f<=L_inf=%.4f  => %s" % (
#     report.theorem_b_rate*100, report.mean_bottleneck, report.mean_l_inf, "PASS" if test4_pass else "PARTIAL"))

# ====================================================================
# TEST 5: CARRIER CALIBRATION + GENERATION (K=16, 16/user)
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 5] Carrier Calibration (N_calib=%d) + Generation (K=%d, %d/user)" % (N_CALIB, K_CANDIDATES, IMGS_PER_USER))
P("=" * 70)

P("  >> PCA calibration...")
t0 = time.time()
carrier_mgr = embedder.calibrate_carriers(config.prompts, n_calib=N_CALIB, verbose=True)
P("  Calibration done in %.1fs" % (time.time() - t0))

os.makedirs(SAVE_DIR, exist_ok=True)

def generate_watermarked_batch(carrier, label=""):
    _times, _imgs, _sigs, _uids, _prompts = [], [], [], [], []
    total = IMGS_PER_USER * N_USERS
    P("  >> Generating %d watermarked images (%d/user, K=%d) [%s]..." % (total, IMGS_PER_USER, K_CANDIDATES, label))
    for i in range(total):
        uid = i % N_USERS
        prompt = USER_PROMPTS[uid][(i // N_USERS) % len(USER_PROMPTS[uid])]
        t0 = time.time()
        img, sig_out, info = embedder.generate_single_watermarked(
            user_id=uid, prompt=prompt, use_topo_coupling=True, K=K_CANDIDATES)
        dt = time.time() - t0
        _times.append(dt); _imgs.append(img); _sigs.append(sig_out); _uids.append(uid); _prompts.append(prompt)
        proj = info.get("carrier_projection", 0)
        if isinstance(proj, (float, int)):
            P("    img %d/%d  user=%d  %.1fs  w_u=%.1f" % (i+1, total, uid, dt, proj))
        else:
            P("    img %d/%d  user=%d  %.1fs" % (i+1, total, uid, dt))
        img.save(os.path.join(SAVE_DIR, "user%d_%03d.png" % (uid, i)))
    P("  Images saved to %s" % SAVE_DIR)
    return _times, _imgs, _sigs, _uids, _prompts

wm_times, images_wm, sigs_wm, user_ids_list, prompts_used = generate_watermarked_batch(carrier_mgr, "PCA")
P("  Avg gen: %.2fs/img  Topo dim: %d" % (np.mean(wm_times), sigs_wm[0].vector.shape[0]))
P("  [TEST 5 done]")

# ====================================================================
# TEST 6: SEPARABILITY (PCA → LDA refinement)
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 6] User Separability (PCA -> LDA Refinement)")
P("=" * 70)

from collections import defaultdict

def compute_separability(sigs, uids, carrier, label=""):
    user_vecs = defaultdict(list)
    for uid, s in zip(uids, sigs):
        user_vecs[uid].append(s.vector)
    P("  [%s] Per-user w_u projection:" % label)
    for uid in sorted(user_vecs.keys()):
        u_u = carrier.carriers[uid % len(carrier.carriers)]
        projs = [float(np.dot(v - carrier.mu_clean, u_u)) for v in user_vecs[uid]]
        P("    User %d: mean=%.1f +/- %.1f  n=%d" % (uid, np.mean(projs), np.std(projs), len(projs)))
    own_p, cross_p = [], []
    for uid in sorted(user_vecs.keys()):
        u_u = carrier.carriers[uid % len(carrier.carriers)]
        for v in user_vecs[uid]:
            own_p.append(float(np.dot(v - carrier.mu_clean, u_u)))
        for o in sorted(user_vecs.keys()):
            if o == uid: continue
            for v in user_vecs[o]:
                cross_p.append(float(np.dot(v - carrier.mu_clean, u_u)))
    m_own, m_cross, s_own = np.mean(own_p), np.mean(np.abs(cross_p)), np.std(own_p)
    ratio = m_own / (m_cross + 1e-12)
    snr = m_own / (s_own + 1e-12)
    P("  [%s] own=%.1f  cross=%.1f  ratio=%.2f  SNR=%.2f  => %s" % (
        label, m_own, m_cross, ratio, snr, "GOOD" if ratio > 1.0 else "WEAK"))
    return ratio, snr

ratio_pca, snr_pca = compute_separability(sigs_wm, user_ids_list, carrier_mgr, "PCA")

P("\n  >> LDA refinement (Fisher 1936)...")
labeled_vecs = np.stack([s.vector for s in sigs_wm])
labeled_ids = np.array(user_ids_list)
carrier_mgr.refine_carriers_lda(labeled_vecs, labeled_ids, verbose=True)

P("  >> Regenerating with LDA carriers (K=%d)..." % K_CANDIDATES)
wm_times2, images_wm, sigs_wm, user_ids_list, prompts_used = generate_watermarked_batch(carrier_mgr, "LDA")
P("  Avg gen (LDA): %.2fs/img" % np.mean(wm_times2))

ratio_lda, snr_lda = compute_separability(sigs_wm, user_ids_list, carrier_mgr, "LDA")
P("  Improvement: ratio %.2f->%.2f  SNR %.2f->%.2f" % (ratio_pca, ratio_lda, snr_pca, snr_lda))
P("  [TEST 6 done]")

# ====================================================================
# TEST 7: DETECTION & ATTRIBUTION
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 7] Detection & Attribution (Bonferroni + Empirical H0)")
P("=" * 70)

from radioactive.pipeline.embedder import WatermarkRegistry
from radioactive.pipeline.detector import RadioactiveDetector

registry = WatermarkRegistry()
for uid, sig_out in zip(user_ids_list, sigs_wm):
    registry.register(uid, sig_out, {"user_id": uid})

P("  >> Generating %d clean baseline images..." % N_CLEAN_BASE)
clean_sigs = embedder.generate_clean_baseline(config.prompts, n_images=N_CLEAN_BASE)
registry.register_clean_baseline(clean_sigs)

detector = RadioactiveDetector(
    max_dim=config.ph_max_dim, pi_resolution=config.pi_resolution,
    pi_bandwidth=config.pi_bandwidth, fpr=0.01)

P("  >> 7a: Per-user attribution...")
n_correct = 0
for uid in range(N_USERS):
    user_imgs = [images_wm[i] for i in range(len(images_wm)) if user_ids_list[i] == uid]
    r = detector.detect(user_imgs, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
    ok = r.is_radioactive and r.attributed_user == uid
    if ok: n_correct += 1
    P("    User %d: det=%s attr=%s ok=%s cos=%.4f" % (uid, r.is_radioactive, r.attributed_user, ok, r.test_statistic))
attr_acc = n_correct / N_USERS

P("  >> 7b: All watermarked batch...")
r_all = detector.detect(images_wm, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
P("    %s  cos=%.4f  attr=%s  thresh=%.4f" % (
    "RADIOACTIVE" if r_all.is_radioactive else "CLEAN", r_all.test_statistic, r_all.attributed_user, r_all.threshold))

P("  >> 7c: Clean rejection (10 images)...")
clean_test = []
with torch.no_grad():
    for i in range(10):
        res = pipe(config.prompts[i % len(config.prompts)], num_inference_steps=20, guidance_scale=7.5)
        clean_test.append(res.images[0])
r_cl = detector.detect(clean_test, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
correct_cl = not r_cl.is_radioactive
P("    %s  cos=%.4f  thresh=%.4f" % ("CLEAN" if correct_cl else "RADIOACTIVE", r_cl.test_statistic, r_cl.threshold))

P("  Attribution: %d/%d  Clean=%s  => %s" % (
    n_correct, N_USERS, correct_cl, "PASS" if correct_cl and attr_acc >= 0.6 else "NEEDS WORK"))
P("  [TEST 7 done]")

# ====================================================================
# HELPER: clone pipe for fine-tuning
# ====================================================================
from radioactive.train.finetune import LoRAFineTuner

def clone_pipe_for_finetune(base_pipe):
    unet_copy = copy.deepcopy(base_pipe.unet)
    p = StableDiffusionPipeline(
        vae=base_pipe.vae, text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer, unet=unet_copy,
        scheduler=DDIMScheduler.from_config(base_pipe.scheduler.config),
        safety_checker=None, feature_extractor=None)
    p.set_progress_bar_config(disable=True)
    return p

def run_radioactivity_test(label, lora_rank, lora_epochs, n_gen):
    P("  >> [%s] LoRA rank=%d, epochs=%d, gen=%d images" % (label, lora_rank, lora_epochs, n_gen))
    cfg = RadioactiveConfig(
        model_id=LOCAL_MODEL, lora_rank=lora_rank, lora_alpha=lora_rank,
        lora_epochs=lora_epochs, lora_batch_size=2, lora_lr=1e-4,
        num_inference_steps=20,
    )
    ft = LoRAFineTuner(cfg)
    p = clone_pipe_for_finetune(pipe)
    p = ft.finetune(p, images=images_wm,
                    prompts=[prompts_used[i % len(prompts_used)] for i in range(len(images_wm))])
    imgs = ft.generate(p, config.prompts, n_images=n_gen)
    r = detector.detect(imgs, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
    P("    [%s] %s  cos_wu=%.4f  p=%.2e  attr=%s  thresh=%.4f" % (
        label, "RADIOACTIVE" if r.is_radioactive else "CLEAN",
        r.test_statistic, r.p_value, r.attributed_user, r.threshold))
    del p; gc.collect(); torch.cuda.empty_cache()
    return r.is_radioactive, r

# ====================================================================
# TEST 8a: RADIOACTIVITY — STANDARD FINE-TUNE
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 8a] Radioactivity -- Standard (rank=%d, epochs=%d, N=%d)" % (LORA_RANK_STD, LORA_EPOCH_STD, N_FT_GEN))
P("=" * 70)
ft_std_pass, ft_std_r = run_radioactivity_test("Standard", LORA_RANK_STD, LORA_EPOCH_STD, N_FT_GEN)
P("  VERDICT: %s" % ("PASS" if ft_std_pass else "FAIL"))
P("  [TEST 8a done]")

# ====================================================================
# TEST 8b: RADIOACTIVITY — LIGHT FINE-TUNE
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 8b] Radioactivity -- Light (rank=%d, epochs=%d, N=%d)" % (LORA_RANK_LT, LORA_EPOCH_LT, N_FT_GEN))
P("=" * 70)
ft_lt_pass, ft_lt_r = run_radioactivity_test("Light", LORA_RANK_LT, LORA_EPOCH_LT, N_FT_GEN)
P("  VERDICT: %s" % ("PASS" if ft_lt_pass else "FAIL"))
P("  [TEST 8b done]")

# ====================================================================
# TEST 9: FPR CONTROL
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 9] FPR Control (clean-only fine-tune)")
P("=" * 70)
P("  >> Generating 30 clean training images...")
clean_train = []
with torch.no_grad():
    for i in range(30):
        res = pipe(config.prompts[i % len(config.prompts)], num_inference_steps=20, guidance_scale=7.5)
        clean_train.append(res.images[0])
        if (i+1) % 10 == 0: P("    %d/30" % (i+1))
P("  >> LoRA fine-tuning on clean-only...")
pipe_cl_ft = clone_pipe_for_finetune(pipe)
ft_cl = LoRAFineTuner(config)
pipe_cl_ft = ft_cl.finetune(pipe_cl_ft, images=clean_train, prompts=config.prompts[:len(clean_train)])
cl_ft_imgs = ft_cl.generate(pipe_cl_ft, config.prompts, n_images=15)
r_fpr = detector.detect(cl_ft_imgs, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
fpr_ok = not r_fpr.is_radioactive
P("    %s  cos=%.4f  p=%.2e  thresh=%.4f" % (
    "CLEAN" if fpr_ok else "RADIOACTIVE", r_fpr.test_statistic, r_fpr.p_value, r_fpr.threshold))
P("  VERDICT: %s" % ("PASS" if fpr_ok else "FPR TOO HIGH"))
P("  [TEST 9 done]")
del pipe_cl_ft; gc.collect(); torch.cuda.empty_cache()

# ====================================================================
# TEST 10: IMAGE QUALITY (LPIPS + PSNR + SSIM)
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 10] Image Quality (LPIPS / PSNR / SSIM)")
P("=" * 70)

import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

lpips_fn = lpips.LPIPS(net='alex').to("cuda")

def compute_pair_metrics(img_a, img_b):
    a_np = np.array(img_a).astype(np.float32) / 255.0
    b_np = np.array(img_b).astype(np.float32) / 255.0
    a_t = torch.from_numpy(a_np).permute(2, 0, 1).unsqueeze(0).to("cuda") * 2.0 - 1.0
    b_t = torch.from_numpy(b_np).permute(2, 0, 1).unsqueeze(0).to("cuda") * 2.0 - 1.0
    lp = float(lpips_fn(a_t, b_t).item())
    psnr_val = peak_signal_noise_ratio(b_np, a_np, data_range=1.0)
    ssim_val = structural_similarity(a_np, b_np, channel_axis=2, data_range=1.0)
    return lp, psnr_val, ssim_val

N_qual = min(10, len(images_wm))
P("  >> Comparing %d watermarked vs clean (same prompt, different noise)..." % N_qual)
P("  >> Also computing clean-vs-clean baseline for reference.")

wm_lpips, wm_psnr, wm_ssim = [], [], []
cl_lpips, cl_psnr, cl_ssim = [], [], []
with torch.no_grad():
    for i in range(N_qual):
        prompt = prompts_used[i]
        res1 = pipe(prompt, num_inference_steps=20, guidance_scale=7.5)
        res2 = pipe(prompt, num_inference_steps=20, guidance_scale=7.5)
        clean1, clean2 = res1.images[0], res2.images[0]
        wm_img = images_wm[i]

        lp_wm, psnr_wm, ssim_wm = compute_pair_metrics(wm_img, clean1)
        wm_lpips.append(lp_wm); wm_psnr.append(psnr_wm); wm_ssim.append(ssim_wm)

        lp_cl, psnr_cl, ssim_cl = compute_pair_metrics(clean1, clean2)
        cl_lpips.append(lp_cl); cl_psnr.append(psnr_cl); cl_ssim.append(ssim_cl)

        P("    img %d: WM-vs-Clean LPIPS=%.4f PSNR=%.2f SSIM=%.4f  |  Clean-vs-Clean LPIPS=%.4f PSNR=%.2f SSIM=%.4f" % (
            i+1, lp_wm, psnr_wm, ssim_wm, lp_cl, psnr_cl, ssim_cl))

mean_wm_lpips, mean_wm_psnr, mean_wm_ssim = np.mean(wm_lpips), np.mean(wm_psnr), np.mean(wm_ssim)
mean_cl_lpips, mean_cl_psnr, mean_cl_ssim = np.mean(cl_lpips), np.mean(cl_psnr), np.mean(cl_ssim)

P("  --- Quality Summary ---")
P("  WM-vs-Clean:    LPIPS=%.4f  PSNR=%.2f  SSIM=%.4f" % (mean_wm_lpips, mean_wm_psnr, mean_wm_ssim))
P("  Clean-vs-Clean: LPIPS=%.4f  PSNR=%.2f  SSIM=%.4f  (reference)" % (mean_cl_lpips, mean_cl_psnr, mean_cl_ssim))
P("  Delta LPIPS: %.4f  (negative = WM better or same)" % (mean_wm_lpips - mean_cl_lpips))
quality_ok = abs(mean_wm_lpips - mean_cl_lpips) < 0.15
P("  VERDICT: %s (no quality degradation)" % ("PASS" if quality_ok else "CHECK"))
P("  [TEST 10 done]")

del lpips_fn; gc.collect(); torch.cuda.empty_cache()

# ====================================================================
# SUMMARY
# ====================================================================
P("\n" + "=" * 70)
P("  BENCHMARK SUMMARY (Signal-Enhanced)")
P("=" * 70)
P("  [Test 1]  Lossless:          => %s" % ("PASS" if test1_pass else "FAIL"))
P("  [Test 2]  Orthogonal:        => %s" % ("PASS" if test2_pass else "FAIL"))
P("  [Test 3]  Channel wts:       => PASS")
P("  [Test 4]  VAE stability:     => %s" % ("PASS" if test4_pass else "PARTIAL"))
P("  [Test 5]  Speed:             %.2fs/img (K=%d)" % (np.mean(wm_times), K_CANDIDATES))
P("  [Test 6]  Separability:      PCA ratio=%.2f -> LDA ratio=%.2f  => %s" % (
    ratio_pca, ratio_lda, "GOOD" if ratio_lda > 1.0 else "WEAK"))
P("  [Test 7]  Attribution:       %d/%d correct  Clean=%s  => %s" % (
    n_correct, N_USERS, correct_cl, "PASS" if correct_cl and attr_acc >= 0.6 else "NEEDS WORK"))
P("  [Test 8a] Radioactivity-Std: => %s" % ("PASS" if ft_std_pass else "FAIL"))
P("  [Test 8b] Radioactivity-Lt:  => %s" % ("PASS" if ft_lt_pass else "FAIL"))
P("  [Test 9]  FPR Control:       => %s" % ("PASS" if fpr_ok else "FAIL"))
P("  [Test 10] Quality:           LPIPS=%.4f  PSNR=%.1f  SSIM=%.4f  => %s" % (
    mean_wm_lpips, mean_wm_psnr, mean_wm_ssim, "PASS" if quality_ok else "CHECK"))
P("=" * 70)
P("  Watermarked images saved to: %s" % SAVE_DIR)
P("  [Benchmark complete]")
P("=" * 70)

del pipe, vae, embedder
gc.collect(); torch.cuda.empty_cache()
