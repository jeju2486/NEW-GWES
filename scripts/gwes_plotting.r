#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Missing R package: data.table.", call. = FALSE)
  }
  library(data.table)
})

print_usage <- function() {
  cat("Usage:\n")
  cat("  Rscript gwes_plotting.r -i stage7.tsv -o out.png -n N [options]\n\n")
  cat("Required:\n")
  cat("  -i, --input            Stage7 TSV (required)\n")
  cat("  -o, --output           Output PNG (required)\n")
  cat("  -n, --num-assemblies   Number of isolates/assemblies (required; PAN-GWES filtering)\n\n")
  cat("Optional:\n")
  cat("  -y, --y-col            y column (default: rMI_e). If srMI_e, uses rMI_e*sign(rlogOR)\n")
  cat("      --n-rows           Read first N rows (default: all)\n")
  cat("  -l, --ld-dist          LD distance line (default: 0)\n")
  cat("      --ld-dist-alt      2nd LD line (default: 0)\n")
  cat("      --stage8           Stage8 TSV for green overlay (default: none)\n")
  cat("      --q-thresh         Stage8 q threshold (default: 0.05)\n")
  cat("      --min-dist         Filter distance >= min-dist (default: 0)\n")
  cat("      --max-points       Downsample background only (default: 0 off)\n")
  cat("      --seed             RNG seed (default: 1)\n")
  cat("      --inblock          Prefix/dir for per-block unitig files (default: none)\n")
  cat("      --mask-cache       RDS cache path (default: auto)\n")
  cat("      --no-deps          Force base plotting (no ggplot2/hexbin)\n")
  cat("      --no-pangwes-filter Skip count/rsd filter\n")
  cat("      --min-count-frac   Default 0.05\n")
  cat("      --rsd-max          Default 1\n")
  cat("  -h, --help             Help\n")
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  o <- list(
    input = NULL, output = NULL, num_assemblies = NULL,
    y_col = "rMI_e", n_rows = 0, ld_dist = 0, ld_dist_alt = 0,
    stage8 = NA_character_, q_thresh = 0.05,
    min_dist = 0, max_points = 0, seed = 1L,
    inblock = NA_character_, mask_cache = NA_character_,
    no_deps = FALSE,
    pangwes_filter = TRUE, min_count_frac = 0.05, rsd_max = 1
  )

  i <- 1
  while (i <= length(args)) {
    a <- args[i]

    if (a %in% c("-h", "--help")) {
      print_usage()
      quit(status = 0)

    } else if (a %in% c("-i", "--input")) {
      o$input <- args[i + 1]
      i <- i + 1

    } else if (a %in% c("-o", "--output")) {
      o$output <- args[i + 1]
      i <- i + 1

    } else if (a %in% c("-n", "--num-assemblies")) {
      o$num_assemblies <- suppressWarnings(as.integer(args[i + 1]))
      i <- i + 1
      if (is.na(o$num_assemblies) || o$num_assemblies < 2) {
        stop("-n must be integer >=2", call. = FALSE)
      }

    } else if (a %in% c("-y", "--y-col")) {
      o$y_col <- args[i + 1]
      i <- i + 1

    } else if (a == "--n-rows") {
      o$n_rows <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1

    } else if (a %in% c("-l", "--ld-dist")) {
      o$ld_dist <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1

    } else if (a == "--ld-dist-alt") {
      o$ld_dist_alt <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1

    } else if (a == "--stage8") {
      o$stage8 <- args[i + 1]
      i <- i + 1

    } else if (a == "--q-thresh") {
      o$q_thresh <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1

    } else if (a == "--min-dist") {
      o$min_dist <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1

    } else if (a == "--max-points") {
      o$max_points <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1

    } else if (a == "--seed") {
      o$seed <- suppressWarnings(as.integer(args[i + 1]))
      i <- i + 1

    } else if (a == "--inblock") {
      o$inblock <- args[i + 1]
      i <- i + 1

    } else if (a == "--mask-cache") {
      o$mask_cache <- args[i + 1]
      i <- i + 1

    } else if (a == "--no-deps") {
      o$no_deps <- TRUE

    } else if (a == "--no-pangwes-filter") {
      o$pangwes_filter <- FALSE

    } else if (a == "--min-count-frac") {
      o$min_count_frac <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1
      if (!is.finite(o$min_count_frac) || o$min_count_frac <= 0 || o$min_count_frac > 1) {
        stop("--min-count-frac in (0,1]", call. = FALSE)
      }

    } else if (a == "--rsd-max") {
      o$rsd_max <- suppressWarnings(as.numeric(args[i + 1]))
      i <- i + 1
      if (!is.finite(o$rsd_max) || o$rsd_max <= 0) {
        stop("--rsd-max > 0", call. = FALSE)
      }

    } else {
      stop(paste0("Unknown option: ", a, " (use --help)"), call. = FALSE)
    }

    i <- i + 1
  }

  if (is.null(o$input) || is.null(o$output) || is.null(o$num_assemblies)) {
    print_usage()
    stop("Missing required: -i -o -n", call. = FALSE)
  }

  o
}

coerce_int_safe <- function(x) {
  xi <- suppressWarnings(as.integer(x))
  if (all(!is.na(xi))) return(xi)
  as.character(x)
}

infer_pair_cols <- function(hdr) {
  if (("unitig_i" %in% hdr) && ("unitig_j" %in% hdr)) return(c("unitig_i", "unitig_j"))
  if (("v" %in% hdr) && ("w" %in% hdr)) return(c("v", "w"))
  stop("Need pair cols (unitig_i,unitig_j) or (v,w).", call. = FALSE)
}

pick_y_col <- function(hdr, y_col) {
  if (y_col %in% hdr) return(y_col)
  cand <- c("srMI_e", "rMI_e", "rMI_10", "rlogOR", "delta11", "MI_obs_e", "MI_null_e")
  pick <- cand[cand %in% hdr]
  if (length(pick) == 0) stop("y_col not found and no fallbacks.", call. = FALSE)
  cat("Requested y_col not found; using ", pick[1], "\n", sep = "")
  pick[1]
}

compute_outlier_thresholds <- function(y) {
  y <- y[is.finite(y)]
  if (length(y) < 10) return(list(out = NA_real_, extreme = NA_real_))
  q <- quantile(y, c(0.25, 0.75), na.rm = TRUE, names = FALSE, type = 7)
  iqr <- q[2] - q[1]
  list(out = q[2] + 1.5 * iqr, extreme = q[2] + 3.0 * iqr)
}

extract_block_idx <- function(fn) {
  bn <- basename(fn)
  m <- regexec("[._]block_([0-9]+)", bn)
  mm <- regmatches(bn, m)
  if (length(mm) >= 1 && length(mm[[1]]) >= 2) return(as.integer(mm[[1]][2]))
  NA_integer_
}

list_per_block_files <- function(inblock_path) {
  if (file.exists(inblock_path) && file.info(inblock_path)$isdir) {
    list.files(inblock_path, pattern = "[._]block_[0-9]+\\.txt$", full.names = TRUE)
  } else {
    unique(c(
      Sys.glob(paste0(inblock_path, ".block_*.txt")),
      Sys.glob(paste0(inblock_path, "_block_*.txt"))
    ))
  }
}

build_mask_cache_path <- function(inblock_path, mask_cache) {
  if (!is.na(mask_cache) && nzchar(mask_cache)) return(mask_cache)
  if (file.exists(inblock_path) && file.info(inblock_path)$isdir) {
    return(file.path(inblock_path, "unitig_blockmask_cache.rds"))
  }
  paste0(inblock_path, ".unitig_blockmask_cache.rds")
}

load_or_build_unitig_masks <- function(inblock_path, cache_path) {
  if (file.exists(cache_path)) return(readRDS(cache_path))
  files <- list_per_block_files(inblock_path)
  if (length(files) == 0) return(data.table(unitig = integer(), mask = integer()))
  else {
    cat("Building unitig masks from", length(files), sep = " ")
  }

  dt_list <- list()
  seen <- integer(0)

  for (fn in files) {
    bidx <- extract_block_idx(fn)
    if (is.na(bidx)) next
    seen <- c(seen, bidx)
    bit <- bitwShiftL(1L, bidx)

    u <- tryCatch(
      fread(fn, header = FALSE, sep = "\n", col.names = "unitig", showProgress = FALSE)[["unitig"]],
      error = function(e) character(0)
    )

    if (length(u) == 0) next

    ui <- suppressWarnings(as.integer(u))
    if (all(!is.na(ui))) {
      dt_list[[length(dt_list) + 1]] <- data.table(unitig = ui, mask = bit)
    } else {
      dt_list[[length(dt_list) + 1]] <- data.table(unitig = as.character(u), mask = bit)
    }
  }

  if (length(dt_list) == 0) return(data.table(unitig = integer(), mask = integer()))

  dt_all <- rbindlist(dt_list)
  dt_map <- dt_all[, .(mask = Reduce(bitwOr, unique(mask))), by = unitig]
  setkey(dt_map, unitig)

  if (length(seen) > 0) attr(dt_map, "max_block_idx") <- max(seen, na.rm = TRUE)
  saveRDS(dt_map, cache_path)
  dt_map
}

opts <- parse_args()

if (!file.exists(opts$input)) stop("Input missing: ", opts$input, call. = FALSE)
if (file.exists(opts$output)) stop("Output exists, delete or change -o: ", opts$output, call. = FALSE)

t0 <- proc.time()[[3]]

hdr <- names(fread(opts$input, nrows = 0L, sep = "\t", showProgress = FALSE))
if (!("distance" %in% hdr)) stop("Missing 'distance' column.", call. = FALSE)

pair_cols <- infer_pair_cols(hdr)
ui_col <- pair_cols[1]
uj_col <- pair_cols[2]
has_missing <- "missing_loci" %in% hdr
count_col <- if ("count" %in% hdr) "count" else NA_character_
m2_col <- if ("M2" %in% hdr) "M2" else if ("d_M2" %in% hdr) "d_M2" else NA_character_

y_col_req <- pick_y_col(hdr, opts$y_col)
need_sr <- identical(y_col_req, "srMI_e")

cols_needed <- c("distance", ui_col, uj_col)
if (has_missing) cols_needed <- c(cols_needed, "missing_loci")
if (opts$pangwes_filter) {
  if (is.na(count_col) || is.na(m2_col)) {
    stop("PAN-GWES filtering needs 'count' and 'M2' (or 'd_M2'). Use --no-pangwes-filter to bypass.", call. = FALSE)
  }
  cols_needed <- c(cols_needed, count_col, m2_col)
}
if (need_sr) {
  cols_needed <- c(cols_needed, "rMI_e", "rlogOR")
} else {
  cols_needed <- c(cols_needed, y_col_req)
}
cols_needed <- unique(cols_needed)

n_to_read <- if (!is.finite(opts$n_rows) || opts$n_rows <= 0) Inf else as.integer(opts$n_rows)
dt7 <- fread(opts$input, sep = "\t", nrows = n_to_read, select = cols_needed, showProgress = TRUE)
cat("Read rows: ", nrow(dt7), "\n", sep = "")

dt7[, distance := suppressWarnings(as.numeric(distance))]
n_disconnected <- sum(is.finite(dt7$distance) & dt7$distance == -1, na.rm = TRUE)

dt7 <- dt7[is.finite(distance)]
dt7 <- dt7[distance != -1]
dt7 <- dt7[distance >= opts$min_dist]

if (has_missing) {
  dt7[, missing_loci := suppressWarnings(as.numeric(missing_loci))]
  dt7 <- dt7[is.finite(missing_loci) & missing_loci == 0]
}

if (opts$pangwes_filter) {
  dt7[, (count_col) := suppressWarnings(as.numeric(get(count_col)))]
  dt7[, (m2_col) := suppressWarnings(as.numeric(get(m2_col)))]
  min_count <- ceiling(opts$min_count_frac * opts$num_assemblies)
  denom <- opts$num_assemblies - 1
  dt7[, rsd := sqrt(pmax(get(m2_col), 0) / denom) / distance]
  dt7[!is.finite(rsd), rsd := Inf]
  before <- nrow(dt7)
  dt7 <- dt7[is.finite(get(count_col)) & get(count_col) >= min_count]
  dt7 <- dt7[is.finite(rsd) & rsd <= opts$rsd_max]
  cat("PAN-GWES kept: ", nrow(dt7), " / ", before, " (min_count=", min_count, ")\n", sep = "")
}

if (nrow(dt7) == 0) stop("No rows after filtering.", call. = FALSE)

if (need_sr) {
  dt7[, rMI_e := suppressWarnings(as.numeric(rMI_e))]
  dt7[, rlogOR := suppressWarnings(as.numeric(rlogOR))]
  dt7 <- dt7[is.finite(rMI_e) & is.finite(rlogOR)]
  dt7[, y := rMI_e * sign(rlogOR)]
} else {
  dt7[, y := suppressWarnings(as.numeric(get(y_col_req)))]
  dt7 <- dt7[is.finite(y)]
}

dt7[, ui := coerce_int_safe(get(ui_col))]
dt7[, uj := coerce_int_safe(get(uj_col))]
dt7[, a := pmin(ui, uj)]
dt7[, b := pmax(ui, uj)]

sig_pairs <- rep(FALSE, nrow(dt7))
if (!is.na(opts$stage8) && nzchar(opts$stage8)) {
  if (!file.exists(opts$stage8)) stop("Stage8 missing: ", opts$stage8, call. = FALSE)
  dt8 <- fread(opts$stage8, sep = "\t", select = c("v", "w", "q_primary"), showProgress = TRUE)
  dt8[, q_primary := suppressWarnings(as.numeric(q_primary))]
  dt8 <- dt8[is.finite(q_primary) & q_primary < opts$q_thresh]
  if (nrow(dt8) > 0) {
    dt8[, v := coerce_int_safe(v)]
    dt8[, w := coerce_int_safe(w)]
    dt8[, a := pmin(v, w)]
    dt8[, b := pmax(v, w)]
    dt8 <- dt8[, .(q = min(q_primary)), by = .(a, b)]
    setkey(dt8, a, b)
    q7 <- dt8[dt7, on = .(a, b), q]
    sig_pairs <- is.finite(q7) & (q7 < opts$q_thresh)
  }
}

highlight_pairs <- rep(FALSE, nrow(dt7))
if (!is.na(opts$inblock) && nzchar(opts$inblock)) {
  cache_path <- build_mask_cache_path(opts$inblock, opts$mask_cache)
  dt_map <- load_or_build_unitig_masks(opts$inblock, cache_path)
  if (nrow(dt_map) > 0) {
    idx1 <- match(dt7$ui, dt_map$unitig)
    idx2 <- match(dt7$uj, dt_map$unitig)
    m1 <- dt_map$mask[idx1]
    m2 <- dt_map$mask[idx2]
    m1[is.na(m1)] <- 0L
    m2[is.na(m2)] <- 0L
    max_block_idx <- attr(dt_map, "max_block_idx")
    if (is.null(max_block_idx) || !is.finite(max_block_idx)) max_block_idx <- -1
    n_pairs <- floor((max_block_idx + 1) / 2)
    if (n_pairs > 0) {
      hp <- rep(FALSE, length(m1))
      for (g in 0:(n_pairs - 1)) {
        bitA <- bitwShiftL(1L, 2 * g)
        bitB <- bitwShiftL(1L, 2 * g + 1)
        hp <- hp |
          ((bitwAnd(m1, bitA) != 0L & bitwAnd(m2, bitB) != 0L) |
           (bitwAnd(m1, bitB) != 0L & bitwAnd(m2, bitA) != 0L))
      }
      highlight_pairs <- hp
    }
  }
}

if (is.finite(opts$max_points) && opts$max_points > 0 && nrow(dt7) > opts$max_points) {
  keep <- sig_pairs | highlight_pairs
  n_keep <- sum(keep)
  if (n_keep >= opts$max_points) {
    sel <- keep
  } else {
    set.seed(opts$seed)
    need <- opts$max_points - n_keep
    bg_idx <- which(!keep)
    take <- sample(bg_idx, size = need, replace = FALSE)
    sel <- keep
    sel[take] <- TRUE
  }
  dt7 <- dt7[sel]
  sig_pairs <- sig_pairs[sel]
  highlight_pairs <- highlight_pairs[sel]
}

distance <- dt7$distance
y <- dt7$y

max_distance <- max(distance, na.rm = TRUE)
if (!is.finite(max_distance) || max_distance <= 0) max_distance <- 1

x_step <- 50000
x_max_tick <- ceiling(max_distance / x_step) * x_step
x_breaks <- seq(0, x_max_tick, by = x_step)

max_abs_y <- max(abs(y[is.finite(y)]), na.rm = TRUE)
if (!is.finite(max_abs_y) || max_abs_y == 0) max_abs_y <- 1
y_lim <- c(-max_abs_y, max_abs_y)
y_ticks <- pretty(y_lim, n = 8)

thr <- compute_outlier_thresholds(y)
outlier_threshold <- thr$out
extreme_outlier_threshold <- thr$extreme

col_bg_blue <- rgb(0, 115, 190, alpha = 60, maxColorValue = 255)
col_sig <- "darkgreen"
col_high <- "red"
col_ld <- "black"
col_ld_alt <- "hotpink1"
col_out <- rgb(165, 0, 38, maxColorValue = 255)
col_ext <- rgb(215, 48, 39, maxColorValue = 255)

plot_width <- 2400
plot_height <- 1200
plot_pointsize <- 16
outdir <- dirname(opts$output)
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

have_ggplot2 <- (!opts$no_deps) &&
  requireNamespace("ggplot2", quietly = TRUE) &&
  requireNamespace("hexbin", quietly = TRUE)

have_ggrastr <- have_ggplot2 && requireNamespace("ggrastr", quietly = TRUE)
have_ggthemes <- have_ggplot2 && requireNamespace("ggthemes", quietly = TRUE)

n_connected_k <- floor(nrow(dt7) / 1000)
n_disconnected_k <- floor(n_disconnected / 1000)

if (have_ggplot2) {
  suppressPackageStartupMessages(library(ggplot2))
  suppressPackageStartupMessages(library(hexbin))
  if (have_ggrastr) suppressPackageStartupMessages(library(ggrastr))
  if (have_ggthemes) suppressPackageStartupMessages(library(ggthemes))

  dtp <- data.table(distance = distance, y = y, sig = sig_pairs, high = highlight_pairs)
  dt_bg <- dtp[!(sig | high)]
  dt_sig <- dtp[sig]
  dt_high <- dtp[high]

  p <- ggplot()

  if (nrow(dt_bg) > 0) {
    layer <- geom_hex(data = dt_bg, aes(x = distance, y = y), bins = 500, fill = col_bg_blue)
    if (have_ggrastr) layer <- ggrastr::rasterise(layer, dpi = 1000)
    p <- p + layer
  }

  p <- p + geom_hline(yintercept = 0, col = "black", linetype = "dashed", linewidth = 0.3)

  if (is.finite(outlier_threshold)) {
    p <- p + geom_hline(yintercept = outlier_threshold, col = col_out, linetype = "dashed", linewidth = 0.3)
  }
  if (is.finite(extreme_outlier_threshold)) {
    p <- p + geom_hline(yintercept = extreme_outlier_threshold, col = col_ext, linetype = "dashed", linewidth = 0.3)
  }

  if (is.finite(opts$ld_dist) && opts$ld_dist > 0) {
    p <- p + geom_vline(xintercept = opts$ld_dist, col = col_ld, linetype = "dashed", linewidth = 0.3)
  }
  if (is.finite(opts$ld_dist_alt) && opts$ld_dist_alt > 0) {
    p <- p + geom_vline(xintercept = opts$ld_dist_alt, col = col_ld_alt, linetype = "dashed", linewidth = 0.3)
  }

  if (nrow(dt_sig) > 0) p <- p + geom_point(data = dt_sig, aes(x = distance, y = y), col = col_sig, pch = 19, size = 0.35)
  if (nrow(dt_high) > 0) p <- p + geom_point(data = dt_high, aes(x = distance, y = y), col = col_high, pch = 19, size = 0.35)

  base_theme <- if (have_ggthemes) ggthemes::theme_clean(base_size = 10) else theme_minimal(base_size = 10)

  p <- p +
    scale_x_continuous(breaks = x_breaks, labels = x_breaks, limits = c(0, x_max_tick)) +
    scale_y_continuous(breaks = y_ticks, limits = y_lim) +
    labs(x = "Distance between unitigs (bp)", y = opts$y_col) +
    base_theme +
    theme(legend.position = "none")

  ann1 <- paste0("#Connected: ", n_connected_k, "K")
  ann2 <- paste0("#Disconnected: ", n_disconnected_k, "K")
  ann3 <- paste0("Stage8 sig: ", sum(sig_pairs), " | True epi: ", sum(highlight_pairs))
  p <- p +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.88 * diff(y_lim), label = ann1, hjust = 1, size = 3.2) +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.84 * diff(y_lim), label = ann2, hjust = 1, size = 3.2) +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.80 * diff(y_lim), label = ann3, hjust = 1, size = 3.2)

  ggplot2::ggsave(
    filename = opts$output,
    plot = p,
    device = "png",
    width = plot_width,
    height = plot_height,
    units = "px"
  )

} else {
  png(opts$output, width = plot_width, height = plot_height, pointsize = plot_pointsize)

  plot(
    distance, y,
    col = col_bg_blue, type = "p", pch = 19, cex = 0.10,
    xlim = c(0, x_max_tick), ylim = y_lim, xaxs = "i", yaxs = "i",
    xlab = "", ylab = "", xaxt = "n", yaxt = "n", bty = "n"
  )

  segments(0, 0, x_max_tick, 0, col = "black", lty = 2, lwd = 1)

  if (is.finite(outlier_threshold)) {
    segments(0, outlier_threshold, x_max_tick, outlier_threshold, col = col_out, lty = 2, lwd = 2)
  }
  if (is.finite(extreme_outlier_threshold)) {
    segments(0, extreme_outlier_threshold, x_max_tick, extreme_outlier_threshold, col = col_ext, lty = 2, lwd = 2)
  }

  if (is.finite(opts$ld_dist) && opts$ld_dist > 0) {
    segments(opts$ld_dist, y_lim[1], opts$ld_dist, y_lim[2], col = col_ld, lty = 2, lwd = 2)
  }
  if (is.finite(opts$ld_dist_alt) && opts$ld_dist_alt > 0) {
    segments(opts$ld_dist_alt, y_lim[1], opts$ld_dist_alt, y_lim[2], col = col_ld_alt, lty = 2, lwd = 2)
  }

  if (any(sig_pairs)) points(distance[sig_pairs], y[sig_pairs], col = col_sig, pch = 19, cex = 0.18)
  if (any(highlight_pairs)) points(distance[highlight_pairs], y[highlight_pairs], col = col_high, pch = 19, cex = 0.18)

  axis(1, at = x_breaks, tick = FALSE, labels = x_breaks, line = -0.8)
  title(xlab = "Distance between unitigs (bp)", line = 1.2)

  axis(2, at = y_ticks, labels = FALSE, tcl = -0.5)
  axis(2, at = y_ticks, labels = y_ticks, las = 1, tcl = -0.5)
  title(ylab = opts$y_col, line = 2.5)

  text(0.90 * x_max_tick, y_lim[1] + 0.88 * diff(y_lim), paste0("#Connected: ", n_connected_k, "K"), cex = 0.8, adj = 0)
  text(0.90 * x_max_tick, y_lim[1] + 0.84 * diff(y_lim), paste0("#Disconnected: ", n_disconnected_k, "K"), cex = 0.8, adj = 0)

  dev.off()
}

cat(sprintf("Wrote: %s\n", opts$output))
cat(sprintf("Total time: %.2fs\n", proc.time()[[3]] - t0))