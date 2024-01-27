# Rain Image Evaluation

* This the codes for evaluation on the metrics (FID, LPIPS, SSIM, PSNR)
* For rain generation results evaluation：
  * Copy the original images to this path：`bash copyDataset.sh`
  * Get the metrics： `bash getMetrics.sh `
* For image rain removal results evaluation：
  * Copy the original images to this path：`bash copyDataset_deraining.sh`
  * Get the metrics for PReNet：`bash getMetrics_deraining_prenet.sh`
  * Get the metrics for M3SNet：`bash getMetrics_deraining_m3snet.sh`
  * Get the metrics for Restormer：`bash getMetrics_deraining_restormer.sh`
  * Get the metrics for SFNet：`bash getMetrics_deraining_sfnet.sh`

