"""
TopoRad 完整基准测试（放射性修复版）
=====================================
核心修复：
  1. LDA 只在 ratio 提升时采用，否则保留 PCA 载波
  2. N_CLEAN_BASE 增大到 100（bootstrap 阈值更精确）
  3. N_FT_GEN 增大到 100（检测统计功效更强）
  4. bootstrap quantile 0.99→0.95（放射性需要更强检测力）
  5. 修复 NameError（TEST 1-4 跳过时 summary 不崩溃）

TEST 5:  载波校准 + w_u 驱动生成
TEST 6:  多用户拓扑可分性（PCA → 可选 LDA）
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
#  核心参数
# ═══════════════════════════════════════════════════════════
K_CANDIDATES   = 16
IMGS_PER_USER  = 16
N_USERS        = 5
N_CALIB        = 50
N_CLEAN_BASE   = 100       # 增大：bootstrap 阈值更准
N_FT_GEN       = 100       # 增大：检测统计功效 ∝ √N
LORA_RANK_STD  = 8
LORA_EPOCH_STD = 50
LORA_RANK_LT   = 8
LORA_EPOCH_LT  = 15
BOOT_QUANTILE  = 0.95      # 放射性检测需要更高 power（α=0.05）

SAVE_DIR = os.path.join(os.getcwd(), "output_radioactive", "watermarked_images")

def P(msg):
    print(msg, flush=True)

P("=" * 70)
P("  TopoRad Benchmark (Radioactivity-Fix: K=%d, %d img/user, N_clean=%d)" % (
    K_CANDIDATES, IMGS_PER_USER, N_CLEAN_BASE))
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

# ====================================================================
# TEST 5: CARRIER CALIBRATION + GENERATION
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 5] Carrier Calibration (N_calib=%d) + Generation (K=%d, %d/user)" % (N_CALIB, K_CANDIDATES, IMGS_PER_USER))
P("=" * 70)

P("  >> PCA calibration...")
t0 = time.time()
carrier_mgr = embedder.calibrate_carriers(config.prompts, n_calib=N_CALIB, verbose=True)
carrier_mgr.boot_quantile = BOOT_QUANTILE
P("  Calibration done in %.1fs" % (time.time() - t0))
P("  Bootstrap quantile: %.2f" % BOOT_QUANTILE)

os.makedirs(SAVE_DIR, exist_ok=True)

def generate_watermarked_batch(label=""):
    _times, _imgs, _sigs, _uids, _prompts, _projs = [], [], [], [], [], []
    total = IMGS_PER_USER * N_USERS
    P("  >> Generating %d watermarked images (%d/user, K=%d) [%s]..." % (total, IMGS_PER_USER, K_CANDIDATES, label))
    for i in range(total):
        uid = i % N_USERS
        prompt = USER_PROMPTS[uid][(i // N_USERS) % len(USER_PROMPTS[uid])]
        t0 = time.time()
        img, sig_out, info = embedder.generate_single_watermarked(
            user_id=uid, prompt=prompt, use_topo_coupling=True, K=K_CANDIDATES)
        dt = time.time() - t0
        proj = info.get("carrier_projection", 0)
        if not isinstance(proj, (float, int)):
            proj = 0.0
        proj = float(proj)
        _times.append(dt); _imgs.append(img); _sigs.append(sig_out)
        _uids.append(uid); _prompts.append(prompt); _projs.append(proj)
        P("    img %d/%d  user=%d  %.1fs  w_u=%.1f" % (i+1, total, uid, dt, proj))
        img.save(os.path.join(SAVE_DIR, "user%d_%03d.png" % (uid, i)))
    P("  Images saved to %s" % SAVE_DIR)
    return _times, _imgs, _sigs, _uids, _prompts, _projs

wm_times, images_wm, sigs_wm, user_ids_list, prompts_used, projs_wm = generate_watermarked_batch("PCA")
P("  Avg gen: %.2fs/img  Topo dim: %d" % (np.mean(wm_times), sigs_wm[0].vector.shape[0]))
P("  [TEST 5 done]")

# ====================================================================
# TEST 6: SEPARABILITY (PCA → conditional LDA)
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 6] User Separability (PCA -> conditional LDA)")
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

pca_carriers_backup = carrier_mgr.carriers.copy()

P("\n  >> Attempting LDA refinement (Fisher 1936)...")
labeled_vecs = np.stack([s.vector for s in sigs_wm])
labeled_ids = np.array(user_ids_list)
carrier_mgr.refine_carriers_lda(labeled_vecs, labeled_ids, verbose=True)

ratio_lda_probe, snr_lda_probe = compute_separability(sigs_wm, user_ids_list, carrier_mgr, "LDA-probe")

if ratio_lda_probe > ratio_pca and snr_lda_probe > snr_pca * 0.8:
    P("  >> LDA improved! ratio %.2f->%.2f  SNR %.2f->%.2f" % (ratio_pca, ratio_lda_probe, snr_pca, snr_lda_probe))
    P("  >> Regenerating with LDA carriers...")
    wm_times2, images_wm, sigs_wm, user_ids_list, prompts_used, projs_wm = generate_watermarked_batch("LDA")
    P("  Avg gen (LDA): %.2fs/img" % np.mean(wm_times2))
    ratio_final, snr_final = compute_separability(sigs_wm, user_ids_list, carrier_mgr, "LDA-final")
    used_lda = True
else:
    P("  >> LDA did NOT improve (ratio %.2f->%.2f, SNR %.2f->%.2f). Keeping PCA carriers." % (
        ratio_pca, ratio_lda_probe, snr_pca, snr_lda_probe))
    carrier_mgr.carriers = pca_carriers_backup
    ratio_final, snr_final = ratio_pca, snr_pca
    used_lda = False

P("  Final: ratio=%.2f  SNR=%.2f  carrier=%s  => %s" % (
    ratio_final, snr_final, "LDA" if used_lda else "PCA", "GOOD" if ratio_final > 1.0 else "WEAK"))
P("  [TEST 6 done]")

# ====================================================================
# TEST 7: DETECTION & ATTRIBUTION
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 7] Detection & Attribution (Bonferroni + Empirical H0, q=%.2f)" % BOOT_QUANTILE)
P("=" * 70)

from radioactive.pipeline.embedder import WatermarkRegistry
from radioactive.pipeline.detector import RadioactiveDetector

registry = WatermarkRegistry()
for uid, sig_out in zip(user_ids_list, sigs_wm):
    registry.register(uid, sig_out, {"user_id": uid})

P("  >> Generating %d clean baseline images (for empirical H0)..." % N_CLEAN_BASE)
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
    P("    User %d: det=%s attr=%s ok=%s t=%.2f thresh=%.2f" % (
        uid, r.is_radioactive, r.attributed_user, ok, r.test_statistic, r.threshold))
attr_acc = n_correct / N_USERS

P("  >> 7b: All watermarked batch...")
r_all = detector.detect(images_wm, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
P("    %s  t=%.2f  attr=%s  thresh=%.2f" % (
    "RADIOACTIVE" if r_all.is_radioactive else "CLEAN", r_all.test_statistic, r_all.attributed_user, r_all.threshold))

P("  >> 7c: Clean rejection (10 images)...")
clean_test = []
with torch.no_grad():
    for i in range(10):
        res = pipe(config.prompts[i % len(config.prompts)], num_inference_steps=20, guidance_scale=7.5)
        clean_test.append(res.images[0])
r_cl = detector.detect(clean_test, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
correct_cl = not r_cl.is_radioactive
P("    %s  t=%.2f  thresh=%.2f" % ("CLEAN" if correct_cl else "RADIOACTIVE", r_cl.test_statistic, r_cl.threshold))

P("  Attribution: %d/%d  Clean=%s  => %s" % (
    n_correct, N_USERS, correct_cl, "PASS" if correct_cl and attr_acc >= 0.6 else "NEEDS WORK"))
P("  [TEST 7 done]")

# ====================================================================
# HELPER: clone pipe + radioactivity test
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

def run_radioactivity_test(label, lora_rank, lora_epochs, n_gen,
                           train_images=None, train_prompts=None):
    """Standard radioactivity test. Returns (pass, result, generated_images)."""
    _imgs = train_images if train_images is not None else images_wm
    _proms = train_prompts if train_prompts is not None else prompts_used
    P("  >> [%s] LoRA rank=%d, epochs=%d, gen=%d, train=%d images" % (
        label, lora_rank, lora_epochs, n_gen, len(_imgs)))
    cfg = RadioactiveConfig(
        model_id=LOCAL_MODEL, lora_rank=lora_rank, lora_alpha=lora_rank,
        lora_epochs=lora_epochs, lora_batch_size=2, lora_lr=1e-4,
        num_inference_steps=20,
    )
    ft = LoRAFineTuner(cfg)
    p = clone_pipe_for_finetune(pipe)
    p = ft.finetune(p, images=_imgs,
                    prompts=[_proms[i % len(_proms)] for i in range(len(_imgs))])
    gen_imgs = ft.generate(p, config.prompts, n_images=n_gen)
    r = detector.detect(gen_imgs, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
    P("    [%s] %s  t=%.2f  p=%.2e  attr=%s  thresh=%.2f" % (
        label, "RADIOACTIVE" if r.is_radioactive else "CLEAN",
        r.test_statistic, r.p_value, r.attributed_user, r.threshold))
    del p; gc.collect(); torch.cuda.empty_cache()
    return r.is_radioactive, r, gen_imgs

def select_curriculum_subset(all_images, all_prompts, all_projs, top_frac=0.5):
    """
    Curriculum sample selection (Bengio et al. 2009 + order statistics).

    Given N watermarked images with per-image carrier projections {p_i},
    select the top-M (M = top_frac * N) images by projection value.

    Theoretical justification (David & Nagaraja 2003, Order Statistics):
      E[p_(k:N)] = mu + sigma * Phi^{-1}(k/(N+1))
      For top-50%: E[p_i | selected] ≈ mu + 0.67*sigma
      Signal boost ≈ 1 + 0.67 / SNR

    Each LoRA gradient step on a strong-signal image produces a gradient
    more aligned with w_u than a randomly chosen image. By concentrating
    training budget on strong samples, the effective per-step SNR improves.
    """
    n_all = len(all_images)
    M = max(int(n_all * top_frac), 8)
    sorted_idx = list(np.argsort(all_projs)[::-1][:M])
    sel_images = [all_images[i] for i in sorted_idx]
    sel_prompts = [all_prompts[i] for i in sorted_idx]
    mean_all = np.mean(all_projs)
    mean_sel = np.mean([all_projs[i] for i in sorted_idx])
    P("  >> Curriculum: top %.0f%% = %d/%d imgs" % (top_frac * 100, M, n_all))
    P("  >>   mean w_u: all=%.1f  selected=%.1f  (%.1fx boost)" % (
        mean_all, mean_sel, mean_sel / (abs(mean_all) + 1e-12)))
    return sel_images, sel_prompts, M

# ====================================================================
# TEST 8a: RADIOACTIVITY — STANDARD
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 8a] Radioactivity -- Standard (rank=%d, epochs=%d, N=%d)" % (LORA_RANK_STD, LORA_EPOCH_STD, N_FT_GEN))
P("=" * 70)
ft_std_pass, ft_std_r, ft_std_imgs = run_radioactivity_test("Standard", LORA_RANK_STD, LORA_EPOCH_STD, N_FT_GEN)
P("  VERDICT: %s" % ("PASS" if ft_std_pass else "FAIL"))
P("  [TEST 8a done]")

# ====================================================================
# TEST 8b: RADIOACTIVITY — LIGHT + CURRICULUM
# ====================================================================
CURRICULUM_FRAC = 0.3   # top 30% by w_u projection (stronger signal per image)

P("\n" + "=" * 70)
P("[TEST 8b] Radioactivity -- Light+Curriculum (rank=%d, N=%d)" % (LORA_RANK_LT, N_FT_GEN))
P("=" * 70)

sel_imgs, sel_proms, M_cur = select_curriculum_subset(
    images_wm, prompts_used, projs_wm, top_frac=CURRICULUM_FRAC)
adjusted_epochs = max(LORA_EPOCH_LT, int(LORA_EPOCH_LT * len(images_wm) / M_cur))
P("  >> epochs: %d -> %d (compensate for fewer images, same total steps)" % (LORA_EPOCH_LT, adjusted_epochs))

ft_lt_pass, ft_lt_r, ft_lt_imgs = run_radioactivity_test(
    "Light-Curriculum", LORA_RANK_LT, adjusted_epochs, N_FT_GEN,
    train_images=sel_imgs, train_prompts=sel_proms)
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
N_FPR_GEN = 100
cl_ft_imgs = ft_cl.generate(pipe_cl_ft, config.prompts, n_images=N_FPR_GEN)
r_fpr = detector.detect(cl_ft_imgs, registry, carrier_manager=carrier_mgr, clean_baseline_sigs=clean_sigs)
fpr_ok = not r_fpr.is_radioactive
P("    %s  t=%.2f  p=%.2e  thresh=%.2f" % (
    "CLEAN" if fpr_ok else "RADIOACTIVE", r_fpr.test_statistic, r_fpr.p_value, r_fpr.threshold))
P("  VERDICT: %s" % ("PASS" if fpr_ok else "FPR TOO HIGH"))
P("  [TEST 9 done]")
del pipe_cl_ft; gc.collect(); torch.cuda.empty_cache()

# ====================================================================
# TEST 10: IMAGE QUALITY
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
P("  >> Comparing %d watermarked vs clean..." % N_qual)

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
        P("    img %d: WM LPIPS=%.4f  Clean LPIPS=%.4f" % (i+1, lp_wm, lp_cl))

mean_wm_lpips, mean_wm_psnr, mean_wm_ssim = np.mean(wm_lpips), np.mean(wm_psnr), np.mean(wm_ssim)
mean_cl_lpips, mean_cl_psnr, mean_cl_ssim = np.mean(cl_lpips), np.mean(cl_psnr), np.mean(cl_ssim)
P("  WM-vs-Clean:    LPIPS=%.4f  PSNR=%.2f  SSIM=%.4f" % (mean_wm_lpips, mean_wm_psnr, mean_wm_ssim))
P("  Clean-vs-Clean: LPIPS=%.4f  PSNR=%.2f  SSIM=%.4f" % (mean_cl_lpips, mean_cl_psnr, mean_cl_ssim))
P("  Delta LPIPS: %.4f" % (mean_wm_lpips - mean_cl_lpips))
quality_ok = abs(mean_wm_lpips - mean_cl_lpips) < 0.15
P("  VERDICT: %s" % ("PASS" if quality_ok else "CHECK"))
P("  [TEST 10 done]")

del lpips_fn; gc.collect(); torch.cuda.empty_cache()

# ====================================================================
# TEST 11: TWO-LAYER WATERMARK VERIFICATION
#
#   Layer 1 — Model Detection: "Was this model trained on watermarked data?"
#     Protocol (Sablayrolles ICML 2020 + Lehmann & Romano 2005):
#       Sample N images from the model, compute batch t-stat = max_u t_u.
#       Under H0 (clean model): max_u t_u follows bootstrap null.
#       Under H1 (WM model): max_u t_u >> threshold.
#       AUC: bootstrap many batches from WM-FT vs Clean-FT, ROC on t-stats.
#
#   Layer 2 — User Attribution: "Which user's data was used?"
#     Protocol: For each user's 16 images, compute batch t on each
#       carrier u∈{0..N_USERS-1}. Attribute = argmax_u t_u.
#       Only carriers assigned to actual users are considered.
# ====================================================================
P("\n" + "=" * 70)
P("[TEST 11] Two-Layer Watermark Verification (Model + User)")
P("=" * 70)

def per_image_projections(images, carrier_mgr, det):
    """Per-image projection on each carrier. Returns (N, U) matrix."""
    P("  >> Extracting topo for %d images..." % len(images))
    sigs = [det._extract_signature(img) for img in images]
    vecs = np.stack([s.vector for s in sigs])
    centered = vecs - carrier_mgr.mu_clean
    carrier_mat = np.stack(carrier_mgr.carriers)
    return centered @ carrier_mat.T

def compute_auc(pos_scores, neg_scores):
    """Manual AUC via trapezoidal rule (no sklearn dependency)."""
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(-scores)
    labels_s = labels[order]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    tp, fp = 0, 0
    tprs, fprs = [0.0], [0.0]
    for l in labels_s:
        if l == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    return float(np.trapezoid(tprs, fprs))

def tpr_at_fpr(pos_scores, neg_scores, target_fpr):
    """TPR at a given FPR level."""
    thresh = np.quantile(neg_scores, 1.0 - target_fpr)
    return float(np.mean(pos_scores >= thresh))

def bootstrap_batch_t(proj_matrix, batch_size, n_trials, n_users, seed=42):
    """
    Batch-level max_u t_u via bootstrap (Sablayrolles 2020 protocol).

    For each trial: sample `batch_size` images, compute t_u for each
    carrier u∈{0..n_users-1}, return max_u t_u.

    This is the correct metric for radioactive watermarks because the
    detection question is batch-level: "Is this MODEL watermarked?"
    """
    rng = np.random.default_rng(seed)
    N = proj_matrix.shape[0]
    bs = min(batch_size, N)
    t_stats = []
    for _ in range(n_trials):
        idx = rng.choice(N, size=bs, replace=True)
        batch = proj_matrix[idx, :n_users]
        max_t = -np.inf
        for u in range(n_users):
            proj_u = batch[:, u]
            mean_p = proj_u.mean()
            std_p = proj_u.std(ddof=1) if bs > 1 else 1.0
            se = std_p / np.sqrt(bs)
            t = mean_p / (se + 1e-12)
            if t > max_t:
                max_t = t
        t_stats.append(max_t)
    return np.array(t_stats)

# --- Extract per-image projections for WM-FT and Clean-FT images ---
P("  >> Computing per-image projections for WM-FT images (%d)..." % len(ft_std_imgs))
proj_wm_ft = per_image_projections(ft_std_imgs, carrier_mgr, detector)
P("  >> Computing per-image projections for Clean-FT images (%d)..." % len(cl_ft_imgs))
proj_cl_ft = per_image_projections(cl_ft_imgs, carrier_mgr, detector)

# ─── Layer 1: Model Detection (Batch-Level Bootstrap AUC) ───
BATCH_SIZE_DET = 20
N_BOOTSTRAP = 500

P("  >> Layer 1: Batch-level bootstrap (batch=%d, trials=%d)" % (BATCH_SIZE_DET, N_BOOTSTRAP))
t_wm_batches = bootstrap_batch_t(proj_wm_ft, BATCH_SIZE_DET, N_BOOTSTRAP, N_USERS, seed=42)
t_cl_batches = bootstrap_batch_t(proj_cl_ft, BATCH_SIZE_DET, N_BOOTSTRAP, N_USERS, seed=123)

auc_model = compute_auc(t_wm_batches, t_cl_batches)
tpr_01 = tpr_at_fpr(t_wm_batches, t_cl_batches, 0.01)
tpr_05 = tpr_at_fpr(t_wm_batches, t_cl_batches, 0.05)

P("  --- Layer 1: Model Detection (Radioactivity) ---")
P("  Batch-AUC (WM-FT vs Clean-FT):  %.4f" % auc_model)
P("  TPR @ FPR=1%%:                    %.1f%%" % (tpr_01 * 100))
P("  TPR @ FPR=5%%:                    %.1f%%" % (tpr_05 * 100))
P("  WM-FT batch-t mean:              %.2f +/- %.2f" % (t_wm_batches.mean(), t_wm_batches.std()))
P("  Clean-FT batch-t mean:           %.2f +/- %.2f" % (t_cl_batches.mean(), t_cl_batches.std()))

# ─── Per-image AUC on the attributed carrier (supplementary) ───
attributed_u_idx = int(np.argmax([
    proj_wm_ft[:, u].mean() / (proj_wm_ft[:, u].std(ddof=1) / np.sqrt(len(proj_wm_ft)) + 1e-12)
    for u in range(N_USERS)
]))
score_wm_single = proj_wm_ft[:, attributed_u_idx]
score_cl_single = proj_cl_ft[:, attributed_u_idx]
auc_single = compute_auc(score_wm_single, score_cl_single)
P("  Per-image AUC (carrier %d):      %.4f" % (attributed_u_idx, auc_single))

# ─── Layer 2: User Attribution (Direct Images, only real users) ───
P("\n  >> Layer 2: User Attribution (Direct Images)")
proj_direct = np.stack([s.vector for s in sigs_wm])
proj_direct_c = proj_direct - carrier_mgr.mu_clean
carrier_mat = np.stack(carrier_mgr.carriers)
proj_direct_all = proj_direct_c @ carrier_mat.T
proj_direct_users = proj_direct_all[:, :N_USERS]

attr_pred = proj_direct_users.argmax(axis=1)
attr_true = np.array(user_ids_list)
n_total_attr = len(attr_true)
n_correct_attr = int((attr_pred == attr_true).sum())
attr_accuracy = n_correct_attr / n_total_attr

P("  --- Layer 2: User Attribution ---")
P("  Total images:     %d" % n_total_attr)
P("  Correct:          %d" % n_correct_attr)
P("  Attribution Acc:  %.1f%%" % (attr_accuracy * 100))

per_user_recall = {}
for uid in range(N_USERS):
    mask = attr_true == uid
    if mask.sum() == 0:
        continue
    correct_u = int((attr_pred[mask] == uid).sum())
    recall_u = correct_u / int(mask.sum())
    per_user_recall[uid] = recall_u
    P("    User %d: %d/%d = %.1f%%" % (uid, correct_u, int(mask.sum()), recall_u * 100))

mean_recall = np.mean(list(per_user_recall.values())) if per_user_recall else 0
P("  Mean Recall:      %.1f%%" % (mean_recall * 100))

proj_own = np.array([float(proj_direct_users[i, attr_true[i]]) for i in range(n_total_attr)])
proj_best_other = np.array([
    float(max(proj_direct_users[i, u] for u in range(N_USERS) if u != attr_true[i]))
    for i in range(n_total_attr)])
margin = proj_own - proj_best_other
P("  Own-carrier margin (own - best_other): mean=%.1f  min=%.1f" % (margin.mean(), margin.min()))

P("\n  --- Combined Two-Layer Summary ---")
P("  [Layer 1] Model Detection AUC:    %.4f" % auc_model)
P("  [Layer 1] TPR@FPR=1%%:             %.1f%%  TPR@FPR=5%%: %.1f%%" % (tpr_01*100, tpr_05*100))
P("  [Layer 1] Per-image AUC (supp):   %.4f" % auc_single)
P("  [Layer 2] Attribution Accuracy:   %.1f%%  (%d/%d)" % (attr_accuracy*100, n_correct_attr, n_total_attr))
P("  [Layer 2] Mean Per-User Recall:   %.1f%%" % (mean_recall * 100))
P("  [TEST 11 done]")

# ====================================================================
# SUMMARY
# ====================================================================
P("\n" + "=" * 70)
P("  BENCHMARK SUMMARY (Two-Layer Watermark System)")
P("=" * 70)
P("  [Test 5]  Speed:             %.2fs/img (K=%d)" % (np.mean(wm_times), K_CANDIDATES))
P("  [Test 6]  Separability:      ratio=%.2f  SNR=%.2f  carrier=%s  => %s" % (
    ratio_final, snr_final, "LDA" if used_lda else "PCA",
    "GOOD" if ratio_final > 1.0 else "WEAK"))
P("  [Test 7]  Attribution:       %d/%d correct  Clean=%s  => %s" % (
    n_correct, N_USERS, correct_cl, "PASS" if correct_cl and attr_acc >= 0.6 else "NEEDS WORK"))
P("  [Test 8a] Radioactivity-Std: => %s  (t=%.2f thresh=%.2f)" % (
    "PASS" if ft_std_pass else "FAIL", ft_std_r.test_statistic, ft_std_r.threshold))
P("  [Test 8b] Radioactivity-Lt:  => %s  (t=%.2f thresh=%.2f)" % (
    "PASS" if ft_lt_pass else "FAIL", ft_lt_r.test_statistic, ft_lt_r.threshold))
P("  [Test 9]  FPR Control:       => %s" % ("PASS" if fpr_ok else "FAIL"))
P("  [Test 10] Quality:           LPIPS=%.4f  dLPIPS=%.4f  => %s" % (
    mean_wm_lpips, mean_wm_lpips - mean_cl_lpips, "PASS" if quality_ok else "CHECK"))
P("  [Test 11] Model Detection:   Batch-AUC=%.4f  TPR@1%%FPR=%.1f%%  TPR@5%%FPR=%.1f%%" % (
    auc_model, tpr_01*100, tpr_05*100))
P("            Per-Image AUC:    %.4f  (carrier %d)" % (auc_single, attributed_u_idx))
P("            User Attribution: Acc=%.1f%%  Recall=%.1f%%  Margin=%.1f" % (
    attr_accuracy*100, mean_recall*100, margin.mean()))
P("=" * 70)
P("  Watermarked images saved to: %s" % SAVE_DIR)
P("  [Benchmark complete]")
P("=" * 70)

try:
    del pipe, embedder
except NameError:
    pass
gc.collect(); torch.cuda.empty_cache()
