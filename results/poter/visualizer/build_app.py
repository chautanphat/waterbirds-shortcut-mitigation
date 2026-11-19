import json
import os

with open("error_samples_summary.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

json_data_str = json.dumps(raw_data)

html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waterbirds Error & Robustness Explorer — POTER vs ERM</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--card, #18181b); }
    ::-webkit-scrollbar-thumb { background: var(--border, #3f3f46); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted-foreground, #71717a); }
    
    .badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 9999px; font-weight: 500; font-size: 12px; }
    .badge-minority { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-majority { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-success { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    
    .tab-btn.active {
      border-bottom: 2px solid var(--primary, #3b82f6);
      color: var(--primary, #3b82f6);
      font-weight: 600;
    }
  </style>
</head>
<body class="bg-[var(--background,#09090b)] text-[var(--foreground,#f4f4f5)] antialiased min-h-screen p-4 sm:p-6 md:p-8">

  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Header Section -->
    <header class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-2xl p-6 shadow-sm">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2 mb-2 flex-wrap">
            <span class="badge badge-success">POTER Benchmark (NeurIPS 2026)</span>
            <span class="badge badge-minority">Waterbirds Dataset</span>
            <span class="badge bg-zinc-800 text-zinc-300 border border-zinc-700">Seed 1 (300 Epochs)</span>
          </div>
          <h1 class="text-2xl sm:text-3xl font-bold tracking-tight text-[var(--foreground,#fafafa)]">
            Waterbirds Error & Robustness Explorer
          </h1>
          <p class="text-sm text-[var(--muted-foreground,#a1a1aa)] mt-1 max-w-3xl">
            Interactive Diagnostic Dashboard analyzing <strong>why shortcut learning fails</strong>, how Optimal Transport Reweighting (POTER) eliminates spurious correlations, and detailed inspection of all 346 residual test errors.
          </p>
        </div>
        <div class="flex items-center gap-3 self-start md:self-auto">
          <div class="text-right">
            <div class="text-xs text-[var(--muted-foreground,#a1a1aa)] uppercase font-semibold">Best Val WGA</div>
            <div class="text-2xl font-black text-emerald-400">90.10%</div>
            <div class="text-xs text-[var(--muted-foreground,#a1a1aa)]">Epoch 258</div>
          </div>
          <div class="h-10 w-px bg-[var(--border,#27272a)]"></div>
          <div class="text-right">
            <div class="text-xs text-[var(--muted-foreground,#a1a1aa)] uppercase font-semibold">Error Reduction</div>
            <div class="text-2xl font-black text-cyan-400">-80.3%</div>
            <div class="text-xs text-[var(--muted-foreground,#a1a1aa)]">vs Standard ERM</div>
          </div>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <nav class="flex border-b border-[var(--border,#27272a)] mt-6 gap-6 text-sm overflow-x-auto">
        <button onclick="switchTab('overview')" id="tab-btn-overview" class="tab-btn active pb-3 transition-colors whitespace-nowrap">
          📊 Tổng Quan & Chỉ Số Lỗi
        </button>
        <button onclick="switchTab('dynamics')" id="tab-btn-dynamics" class="tab-btn pb-3 transition-colors text-[var(--muted-foreground,#a1a1aa)] whitespace-nowrap">
          📈 Tiến Trình 300 Epochs (Interactive Scrubber)
        </button>
        <button onclick="switchTab('causes')" id="tab-btn-causes" class="tab-btn pb-3 transition-colors text-[var(--muted-foreground,#a1a1aa)] whitespace-nowrap">
          🔍 Tại Sao Mô Hình Sai? (Root Cause Analysis)
        </button>
        <button onclick="switchTab('samples')" id="tab-btn-samples" class="tab-btn pb-3 transition-colors text-[var(--muted-foreground,#a1a1aa)] whitespace-nowrap">
          📋 Tra Cứu Từng Ca Lỗi (Error Browser: 346 Cases)
        </button>
        <button onclick="switchTab('matrix')" id="tab-btn-matrix" class="tab-btn pb-3 transition-colors text-[var(--muted-foreground,#a1a1aa)] whitespace-nowrap">
          ⚖️ Đối Chiếu ERM vs POTER
        </button>
      </nav>
    </header>

    <!-- TAB 1: OVERVIEW -->
    <div id="tab-overview" class="space-y-6">
      <!-- KPI Metric Cards Grid -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <!-- Card 1 -->
        <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-5 shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--muted-foreground,#a1a1aa)] mb-1">
            <span>Worst-Group Accuracy</span>
            <span class="badge badge-success">+55.2% vs ERM</span>
          </div>
          <div class="text-2xl font-extrabold text-[var(--foreground,#fafafa)]">90.10%</div>
          <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mt-1">
            Khắc phục nhóm khó nhất (Group 1: Chim cạn trên nền nước). ERM chỉ đạt <strong>34.90%</strong>.
          </p>
        </div>

        <!-- Card 2 -->
        <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-5 shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--muted-foreground,#a1a1aa)] mb-1">
            <span>Tổng Số Lỗi Test (5.794 ảnh)</span>
            <span class="badge badge-success">-1.406 lỗi</span>
          </div>
          <div class="text-2xl font-extrabold text-[var(--foreground,#fafafa)]">346 <span class="text-xs font-normal text-[var(--muted-foreground,#a1a1aa)]">/ 5.794 (5.97%)</span></div>
          <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mt-1">
            ERM sai tới <strong>1.752 ảnh (30.24%)</strong> do bị bẫy bởi phông nền nước và cạn.
          </p>
        </div>

        <!-- Card 3 -->
        <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-5 shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--muted-foreground,#a1a1aa)] mb-1">
            <span>Cứu Nhóm Thiểu Số Group 1</span>
            <span class="badge badge-minority">Tương quan giả</span>
          </div>
          <div class="text-2xl font-extrabold text-amber-400">176 lỗi <span class="text-xs font-normal text-[var(--muted-foreground,#a1a1aa)]">vs 1.468 lỗi ERM</span></div>
          <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mt-1">
            Giảm <strong>88.0% số ca lỗi</strong> trên nhóm chim cạn bị ghép vào phông nước.
          </p>
        </div>

        <!-- Card 4 -->
        <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-5 shadow-sm">
          <div class="flex items-center justify-between text-xs text-[var(--muted-foreground,#a1a1aa)] mb-1">
            <span>Độ Tự Tin Lỗi Sai (Confidence)</span>
            <span class="badge badge-success">Hiệu chuẩn tốt</span>
          </div>
          <div class="text-2xl font-extrabold text-[var(--foreground,#fafafa)]">4.9% <span class="text-xs font-normal text-[var(--muted-foreground,#a1a1aa)]">overconfident (&ge;90%)</span></div>
          <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mt-1">
            ERM đoán sai một cách mù quáng với độ tự tin &ge;90% ở <strong>23.3%</strong> số ca lỗi.
          </p>
        </div>
      </div>

      <!-- 4 Groups Comparison Grid -->
      <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-6 shadow-sm">
        <h3 class="text-lg font-semibold text-[var(--foreground,#fafafa)] mb-1">
          Bản Đồ Phân Phối Độ Chính Xác & Lỗi Trên 4 Nhóm Dữ Liệu
        </h3>
        <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mb-6">
          Đối chiếu trực tiếp giữa mô hình thường (ERM) và mô hình tối ưu trọng số (POTER). Nhóm thiểu số (Minority) là nơi bộc lộ rõ nhất sự khác biệt.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Group 0 -->
          <div class="border border-[var(--border,#27272a)] rounded-lg p-4 bg-zinc-900/50">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-blue-500"></span>
                <span class="font-semibold text-sm">Group 0: Chim cạn trên nền cạn (Đa số)</span>
              </div>
              <span class="badge badge-majority">Majority: 3.498 train (72.9%)</span>
            </div>
            <div class="space-y-2 mt-3">
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>POTER Accuracy: <strong>92.50%</strong> (35 lỗi / 467 val)</span>
                  <span class="text-emerald-400 font-semibold">92.5%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-emerald-500 h-2.5 rounded-full" style="width: 92.5%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>ERM Accuracy: <strong>99.60%</strong> (9 lỗi / 2.255 test)</span>
                  <span class="text-blue-400 font-semibold">99.6%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-blue-500 h-2.5 rounded-full" style="width: 99.6%"></div>
                </div>
              </div>
            </div>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mt-3">
              ERM đạt điểm gần như tuyệt đối ở nhóm này vì tận dụng cả 2 đặc trưng: vừa có hình chim cạn, vừa có nền cạn hỗ trợ.
            </p>
          </div>

          <!-- Group 1 -->
          <div class="border border-red-500/30 rounded-lg p-4 bg-red-950/10">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-red-500"></span>
                <span class="font-semibold text-sm text-red-300">Group 1: Chim cạn trên nền nước (Bẫy tương quan giả khó nhất)</span>
              </div>
              <span class="badge badge-minority">Minority: 184 train (3.8%)</span>
            </div>
            <div class="space-y-2 mt-3">
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>POTER Accuracy: <strong>90.10%</strong> (46 lỗi / 466 val)</span>
                  <span class="text-emerald-400 font-semibold">90.1%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-emerald-500 h-2.5 rounded-full" style="width: 90.1%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>ERM Accuracy: <strong>34.90%</strong> (1.468 lỗi / 2.255 test)</span>
                  <span class="text-red-400 font-semibold">34.9% (SỤP ĐỔ)</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-red-500 h-2.5 rounded-full" style="width: 34.9%"></div>
                </div>
              </div>
            </div>
            <p class="text-xs text-red-200/80 mt-3 font-medium">
              🔥 Điểm cốt tử: ERM dự đoán sai tới 65.1% số ảnh vì nền nước lấn át đặc trưng con chim. POTER nâng vọt độ chính xác lên 90.10%!
            </p>
          </div>

          <!-- Group 2 -->
          <div class="border border-amber-500/30 rounded-lg p-4 bg-amber-950/10">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-amber-500"></span>
                <span class="font-semibold text-sm text-amber-300">Group 2: Chim nước trên nền cạn (Nhóm siêu hiếm)</span>
              </div>
              <span class="badge badge-warning">Cực hiếm: 56 train (1.17%)</span>
            </div>
            <div class="space-y-2 mt-3">
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>POTER Accuracy: <strong>92.50%</strong> (10 lỗi / 133 val)</span>
                  <span class="text-emerald-400 font-semibold">92.5%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-emerald-500 h-2.5 rounded-full" style="width: 92.5%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>ERM Accuracy: <strong>60.12%</strong> (256 lỗi / 642 test)</span>
                  <span class="text-amber-400 font-semibold">60.1%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-amber-500 h-2.5 rounded-full" style="width: 60.1%"></div>
                </div>
              </div>
            </div>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mt-3">
              Chỉ có 56 mẫu huấn luyện nhưng POTER vẫn đạt 92.50% nhờ cơ chế Optimal Transport gán trọng số cực cao cho nhóm này.
            </p>
          </div>

          <!-- Group 3 -->
          <div class="border border-[var(--border,#27272a)] rounded-lg p-4 bg-zinc-900/50">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-cyan-500"></span>
                <span class="font-semibold text-sm">Group 3: Chim nước trên nền nước (Đa số)</span>
              </div>
              <span class="badge badge-majority">Majority: 1.057 train (22.0%)</span>
            </div>
            <div class="space-y-2 mt-3">
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>POTER Accuracy: <strong>92.50%</strong> (10 lỗi / 133 val)</span>
                  <span class="text-emerald-400 font-semibold">92.5%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-emerald-500 h-2.5 rounded-full" style="width: 92.5%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-xs mb-1">
                  <span>ERM Accuracy: <strong>97.04%</strong> (19 lỗi / 642 test)</span>
                  <span class="text-cyan-400 font-semibold">97.0%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-2.5">
                  <div class="bg-cyan-500 h-2.5 rounded-full" style="width: 97.0%"></div>
                </div>
              </div>
            </div>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] mt-3">
              Nhóm tương quan thuận, cả hai mô hình đều nhận diện chuẩn xác.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: DYNAMICS (300 EPOCHS SCRUBBER) -->
    <div id="tab-dynamics" class="space-y-6 hidden">
      <!-- Scrubber Control Card -->
      <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-5 shadow-sm space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 class="font-semibold text-base text-[var(--foreground,#fafafa)]">
              Điều Khiển Thời Gian Huấn Luyện (Interactive Epoch Scrubber)
            </h3>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)]">
              Kéo thanh trượt hoặc nhấn Play để xem mô hình học cách xóa bỏ tương quan giả qua 300 epochs.
            </p>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="stepEpoch(-1)" class="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg border border-zinc-700">
              ⏮ Lùi 1
            </button>
            <button onclick="togglePlay()" id="play-btn" class="px-4 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg shadow-sm">
              ▶ Chạy (Play)
            </button>
            <button onclick="stepEpoch(1)" class="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg border border-zinc-700">
              Tiến 1 ⏭
            </button>
          </div>
        </div>

        <!-- Slider Bar -->
        <div class="space-y-2">
          <div class="flex justify-between items-center text-xs flex-wrap gap-2">
            <span class="text-[var(--muted-foreground,#a1a1aa)]">Epoch: <strong id="current-epoch-label" class="text-blue-400 text-base font-black">258</strong> / 299</span>
            <div class="flex gap-2 flex-wrap">
              <button onclick="jumpToEpoch(0)" class="text-xs text-zinc-400 hover:text-white underline">Epoch 0 (Đầu)</button>
              <button onclick="jumpToEpoch(50)" class="text-xs text-zinc-400 hover:text-white underline">Epoch 50</button>
              <button onclick="jumpToEpoch(100)" class="text-xs text-zinc-400 hover:text-white underline">Epoch 100</button>
              <button onclick="jumpToEpoch(258)" class="text-xs text-emerald-400 font-bold hover:underline">Epoch 258 (Best WGA)</button>
              <button onclick="jumpToEpoch(299)" class="text-xs text-zinc-400 hover:text-white underline">Epoch 299 (Cuối)</button>
            </div>
          </div>
          <input type="range" id="epoch-slider" min="0" max="299" value="258" oninput="onEpochChange(this.value)" class="w-full accent-blue-500 cursor-pointer">
        </div>

        <!-- Live Snapshot Grid for Selected Epoch -->
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-2">
          <div class="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg text-center">
            <div class="text-[11px] text-zinc-400">Val Worst-Group (WGA)</div>
            <div id="stat-val-wga" class="text-xl font-black text-emerald-400">90.1%</div>
          </div>
          <div class="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg text-center">
            <div class="text-[11px] text-zinc-400">Val Avg Accuracy</div>
            <div id="stat-val-avg" class="text-xl font-bold text-blue-400">91.6%</div>
          </div>
          <div class="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg text-center">
            <div class="text-[11px] text-zinc-400">Group 1 (Chim cạn/Nước)</div>
            <div id="stat-g1" class="text-xl font-bold text-amber-400">90.1%</div>
          </div>
          <div class="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg text-center">
            <div class="text-[11px] text-zinc-400">Group 2 (Chim nước/Cạn)</div>
            <div id="stat-g2" class="text-xl font-bold text-purple-400">92.5%</div>
          </div>
          <div class="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg text-center">
            <div class="text-[11px] text-zinc-400">Train Loss / Val Loss</div>
            <div id="stat-loss" class="text-sm font-semibold text-zinc-200 mt-1">0.163 / 0.292</div>
          </div>
          <div class="bg-zinc-900/60 border border-zinc-800 p-2.5 rounded-lg text-center">
            <div class="text-[11px] text-zinc-400">Trạng Thái Huấn Luyện</div>
            <div id="stat-phase" class="text-xs font-semibold text-emerald-400 mt-1">Đỉnh Tối Ưu</div>
          </div>
        </div>
      </div>

      <!-- Charts Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Chart 1: WGA & Loss Trajectory -->
        <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-5 shadow-sm">
          <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
            <div>
              <h4 class="font-semibold text-sm text-[var(--foreground,#fafafa)]">Quỹ Đạo WGA & Loss (300 Epochs)</h4>
              <p class="text-xs text-[var(--muted-foreground,#a1a1aa)]">Đường xanh lá: Validation WGA (Đỉnh 90.1% ở Epoch 258)</p>
            </div>
            <div class="flex items-center gap-3 text-xs flex-wrap">
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span> Val WGA</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-blue-400"></span> Val Avg</span>
            </div>
          </div>
          <div class="relative w-full h-64 bg-zinc-950/40 rounded-lg p-2 border border-zinc-800/80">
            <canvas id="chart-wga-loss" class="w-full h-full block"></canvas>
          </div>
        </div>

        <!-- Chart 2: Per-Group Accuracy Trajectory -->
        <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-5 shadow-sm">
          <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
            <div>
              <h4 class="font-semibold text-sm text-[var(--foreground,#fafafa)]">Độ Chính Xác Từng Nhóm Qua 300 Epochs</h4>
              <p class="text-xs text-[var(--muted-foreground,#a1a1aa)]">Quan sát đường Group 1 (đỏ) vọt từ 29.6% lên 90.1%</p>
            </div>
            <div class="flex items-center gap-3 text-xs flex-wrap">
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-blue-400"></span> G0</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-red-400"></span> G1 (Bẫy)</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span> G2</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-cyan-400"></span> G3</span>
            </div>
          </div>
          <div class="relative w-full h-64 bg-zinc-950/40 rounded-lg p-2 border border-zinc-800/80">
            <canvas id="chart-groups" class="w-full h-full block"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: ROOT CAUSE ANALYSIS (TẠI SAO SAI?) -->
    <div id="tab-causes" class="space-y-6 hidden">
      <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-6 shadow-sm">
        <h3 class="text-xl font-bold text-[var(--foreground,#fafafa)] mb-2">
          Tại Sao Mô Hình Bị Sai? — 4 Nguyên Nhân Gốc Rễ
        </h3>
        <p class="text-sm text-[var(--muted-foreground,#a1a1aa)] max-w-4xl mb-6">
          Phân tích sâu về mặt thị giác máy tính và thống kê học để giải thích chính xác tại sao mô hình gặp sai sót trên tập dữ liệu Waterbirds.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Cause 1 -->
          <div class="border border-red-500/20 bg-red-950/10 rounded-xl p-5 space-y-3">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-lg bg-red-500/20 text-red-400 flex items-center justify-center font-bold text-lg">1</div>
              <div>
                <h4 class="font-bold text-sm text-[var(--foreground,#fafafa)]">Bẫy Tương Quan Giả Phông Nền (Background Shortcut Dominance)</h4>
                <div class="text-xs text-red-400 font-medium">Chiếm > 50% số ca lỗi</div>
              </div>
            </div>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] leading-relaxed">
              Trong tập huấn luyện, <strong>95% ảnh có nước là chim nước</strong>, và <strong>95% ảnh có rừng là chim cạn</strong>. Vì mạng ResNet-50 trích xuất đặc trưng toàn cục, diện tích phông nền (nước, sóng biển, cây rừng) chiếm tới <strong>80–90% diện tích bức ảnh</strong>.
            </p>
            <div class="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-xs space-y-1">
              <div class="text-zinc-300 font-semibold">Triệu chứng lâm sàng:</div>
              <div class="text-zinc-400">• ERM dự đoán nhầm 1.468 / 2.255 ảnh chim cạn trên nước thành chim nước vì nhìn thấy màu xanh nước biển.</div>
              <div class="text-zinc-400">• POTER đã giải quyết được 88% bẫy này nhờ Optimal Transport gán trọng số lớn cho nhóm đối nghịch.</div>
            </div>
          </div>

          <!-- Cause 2 -->
          <div class="border border-amber-500/20 bg-amber-950/10 rounded-xl p-5 space-y-3">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-lg">2</div>
              <div>
                <h4 class="font-bold text-sm text-[var(--foreground,#fafafa)]">Ngoại Lệ Sinh Học & Môi Trường (Ecological Ambiguity)</h4>
                <div class="text-xs text-amber-400 font-medium">Các loài đặc biệt như Kingfisher, Fish Crow, Pewee</div>
              </div>
            </div>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] leading-relaxed">
              Theo cây phân loại sinh học CUB, chim Bói Cá (<strong>Belted Kingfisher, White-breasted Kingfisher</strong>) thuộc bộ Sả (Coraciiformes) - được gán nhãn là <em>Landbird</em>. Tuy nhiên, tập tính tự nhiên của bói cá là đậu cành cây sát mặt nước hoặc bổ nhào xuống nước bắt cá, có mỏ dài nhọn giống chim biển.
            </p>
            <div class="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-xs space-y-1">
              <div class="text-zinc-300 font-semibold">Hậu quả thực tế:</div>
              <div class="text-zinc-400">• Kingfisher có tới 18 ca lỗi trong tập test (nằm trong top 5 loài sai nhiều nhất).</div>
              <div class="text-zinc-400">• Mô hình bị bối rối vì con chim mang các đặc trưng thị giác của cả 2 lớp.</div>
            </div>
          </div>

          <!-- Cause 3 -->
          <div class="border border-blue-500/20 bg-blue-950/10 rounded-xl p-5 space-y-3">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold text-lg">3</div>
              <div>
                <h4 class="font-bold text-sm text-[var(--foreground,#fafafa)]">Tỷ Lệ Kích Thước Quá Nhỏ & Ngụy Trang (Scale & Camouflage)</h4>
                <div class="text-xs text-blue-400 font-medium">Chủ thể chiếm &lt; 5% số lượng điểm ảnh</div>
              </div>
            </div>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] leading-relaxed">
              Nhiều bức ảnh chụp từ khoảng cách xa: con chim chỉ là một chấm nhỏ mờ giữa rừng cây rậm rạp hoặc mặt nước mênh mông. Lớp lông chim tiệp màu với thân cây hoặc bóng râm khiến mạng nơ-ron không tách được vùng quan tâm (ROI).
            </p>
            <div class="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-xs space-y-1">
              <div class="text-zinc-300 font-semibold">Ví dụ điển hình:</div>
              <div class="text-zinc-400">• Loài <em>Western Wood Pewee</em> và <em>Eastern Towhee</em>: lông xám nâu lẫn hoàn toàn vào cành cây khô, dẫn tới 47 ca lỗi.</div>
            </div>
          </div>

          <!-- Cause 4 -->
          <div class="border border-purple-500/20 bg-purple-950/10 rounded-xl p-5 space-y-3">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold text-lg">4</div>
              <div>
                <h4 class="font-bold text-sm text-[var(--foreground,#fafafa)]">Bản Chất Của Lỗi: Lưỡng Lự Chứ Không Mù Quáng</h4>
                <div class="text-xs text-purple-400 font-medium">Borderline Uncertainty (Độ tự tin trung bình 67.4%)</div>
              </div>
            </div>
            <p class="text-xs text-[var(--muted-foreground,#a1a1aa)] leading-relaxed">
              Điểm tiến bộ lớn nhất của POTER so với ERM là <strong>độ hiệu chuẩn xác suất (calibration)</strong>:
              Khi POTER đoán đúng, độ tự tin đạt trung bình <strong>84.3%</strong>. Khi đoán sai, độ tự tin tụt xuống <strong>67.4%</strong> (vùng lưỡng lự 50-70%).
            </p>
            <div class="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-xs space-y-1">
              <div class="text-zinc-300 font-semibold">Khác biệt với ERM:</div>
              <div class="text-zinc-400">• ERM đoán sai với độ tự tin cực cao (&ge; 90%) ở 23.3% số ca lỗi.</div>
              <div class="text-zinc-400">• POTER chỉ có 4.9% lỗi là overconfident, 95.1% còn lại là các ca khó thực sự.</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: SAMPLE EXPLORER (346 ERROR CASES) -->
    <div id="tab-samples" class="space-y-6 hidden">
      <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-6 shadow-sm space-y-4">
        <div>
          <h3 class="text-lg font-bold text-[var(--foreground,#fafafa)]">
            Tra Cứu Chi Tiết Từng Ca Lỗi (Error Sample Browser)
          </h3>
          <p class="text-xs text-[var(--muted-foreground,#a1a1aa)]">
            Lọc và tìm kiếm danh sách 346 ca dự đoán sai của mô hình trên tập kiểm thử Waterbirds.
          </p>
        </div>

        <!-- Filter & Search Controls -->
        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 pt-2">
          <div>
            <label class="block text-xs font-semibold text-[var(--muted-foreground,#a1a1aa)] mb-1">Tìm theo tên loài chim / File</label>
            <input type="text" id="search-input" placeholder="VD: Kingfisher, Pewee..." oninput="applyFilters()" class="w-full text-xs bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500">
          </div>

          <div>
            <label class="block text-xs font-semibold text-[var(--muted-foreground,#a1a1aa)] mb-1">Lọc theo nhóm (Group)</label>
            <select id="group-filter" onchange="applyFilters()" class="w-full text-xs bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500">
              <option value="all">Tất cả nhóm (346 lỗi)</option>
              <option value="Landbird on Water">Group 1: Landbird on Water (176 lỗi)</option>
              <option value="Landbird on Land">Group 0: Landbird on Land (75 lỗi)</option>
              <option value="Waterbird on Land">Group 2: Waterbird on Land (49 lỗi)</option>
              <option value="Waterbird on Water">Group 3: Waterbird on Water (46 lỗi)</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-semibold text-[var(--muted-foreground,#a1a1aa)] mb-1">Mức độ tự tin (Confidence)</label>
            <select id="conf-filter" onchange="applyFilters()" class="w-full text-xs bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500">
              <option value="all">Tất cả mức tự tin</option>
              <option value="high">Cao (Overconfident &ge; 0.85)</option>
              <option value="medium">Trung bình (0.65 - 0.85)</option>
              <option value="low">Lưỡng lự (Borderline &lt; 0.65)</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-semibold text-[var(--muted-foreground,#a1a1aa)] mb-1">So sánh với ERM</label>
            <select id="erm-filter" onchange="applyFilters()" class="w-full text-xs bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500">
              <option value="all">Tất cả</option>
              <option value="both_wrong">Cả ERM và POTER đều sai</option>
              <option value="erm_wrong_high">ERM sai với tự tin &ge; 0.80</option>
            </select>
          </div>
        </div>

        <div class="text-xs text-[var(--muted-foreground,#a1a1aa)] flex justify-between items-center pt-1">
          <span>Tìm thấy <strong id="filter-count" class="text-white">346</strong> ca lỗi phù hợp</span>
          <span>Hiển thị trang <span id="current-page">1</span> / <span id="total-pages">24</span></span>
        </div>

        <!-- Error Table -->
        <div class="overflow-x-auto border border-zinc-800 rounded-lg">
          <table class="w-full text-left text-xs text-zinc-300">
            <thead class="bg-zinc-900 text-zinc-400 uppercase font-semibold text-[11px] border-b border-zinc-800">
              <tr>
                <th class="p-3">Loài Chim / Tên File</th>
                <th class="p-3">Nhóm Dữ Liệu</th>
                <th class="p-3">Thực Tế &rarr; Dự Đoán</th>
                <th class="p-3">Độ Tự Tin</th>
                <th class="p-3">Mô Hình ERM</th>
                <th class="p-3">Chẩn Đoán Nguyên Nhân</th>
              </tr>
            </thead>
            <tbody id="error-table-body" class="divide-y divide-zinc-800">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>

        <!-- Pagination -->
        <div class="flex items-center justify-between pt-2">
          <button onclick="changePage(-1)" id="prev-page-btn" class="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg border border-zinc-700">
            &larr; Trang trước
          </button>
          <div class="text-xs text-zinc-400">
            Trang <strong id="page-indicator" class="text-white">1</strong>
          </div>
          <button onclick="changePage(1)" id="next-page-btn" class="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg border border-zinc-700">
            Trang sau &rarr;
          </button>
        </div>
      </div>
    </div>

    <!-- TAB 5: ERM VS POTER MATRIX -->
    <div id="tab-matrix" class="space-y-6 hidden">
      <div class="bg-[var(--card,#18181b)] border border-[var(--border,#27272a)] rounded-xl p-6 shadow-sm space-y-6">
        <div>
          <h3 class="text-xl font-bold text-[var(--foreground,#fafafa)] mb-1">
            Bảng Đối Chiếu Ma Trận Lỗi: ERM vs POTER
          </h3>
          <p class="text-xs text-[var(--muted-foreground,#a1a1aa)]">
            Phân tích số lượng ca đúng / sai trên toàn bộ 5.794 ảnh của tập kiểm thử.
          </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- ERM Confusion Matrix -->
          <div class="border border-zinc-800 rounded-xl p-4 bg-zinc-900/40 space-y-3">
            <div class="flex items-center justify-between">
              <span class="font-bold text-sm text-red-400">Mô hình thường (Standard ERM)</span>
              <span class="badge badge-minority">WGA: 34.90%</span>
            </div>
            <p class="text-xs text-zinc-400">Tổng số lỗi: <strong>1.752 / 5.794 (Sai 30.2%)</strong></p>
            <div class="grid grid-cols-2 gap-2 text-center text-xs">
              <div class="p-3 bg-zinc-800/80 rounded border border-zinc-700">
                <div class="text-zinc-400 text-[11px]">Đoán đúng Chim Cạn</div>
                <div class="text-lg font-bold text-blue-400">3.033</div>
              </div>
              <div class="p-3 bg-red-950/40 rounded border border-red-800/50">
                <div class="text-red-300 text-[11px]">Cạn nhầm thành Nước</div>
                <div class="text-lg font-bold text-red-400">1.477</div>
                <div class="text-[10px] text-red-300">1.468 ca do phông nước</div>
              </div>
              <div class="p-3 bg-amber-950/40 rounded border border-amber-800/50">
                <div class="text-amber-300 text-[11px]">Nước nhầm thành Cạn</div>
                <div class="text-lg font-bold text-amber-400">275</div>
                <div class="text-[10px] text-amber-300">256 ca do phông cạn</div>
              </div>
              <div class="p-3 bg-zinc-800/80 rounded border border-zinc-700">
                <div class="text-zinc-400 text-[11px]">Đoán đúng Chim Nước</div>
                <div class="text-lg font-bold text-cyan-400">1.009</div>
              </div>
            </div>
            <div class="p-2.5 bg-red-950/20 border border-red-900/30 rounded text-xs text-red-200">
              ⚠️ <strong>Kết luận ERM:</strong> Mô hình học thành công phông nền thay vì học con chim. Nếu nền là nước, nó đoán chim nước; nếu nền là cạn, nó đoán chim cạn.
            </div>
          </div>

          <!-- POTER Confusion Matrix -->
          <div class="border border-emerald-500/30 rounded-xl p-4 bg-emerald-950/10 space-y-3">
            <div class="flex items-center justify-between">
              <span class="font-bold text-sm text-emerald-400">Mô hình POTER (Optimal Transport)</span>
              <span class="badge badge-success">Val WGA: 90.10%</span>
            </div>
            <p class="text-xs text-zinc-400">Tổng số lỗi: <strong>346 / 5.794 (Sai chỉ 5.97%)</strong></p>
            <div class="grid grid-cols-2 gap-2 text-center text-xs">
              <div class="p-3 bg-zinc-800/80 rounded border border-zinc-700">
                <div class="text-zinc-400 text-[11px]">Đoán đúng Chim Cạn</div>
                <div class="text-lg font-bold text-emerald-400">4.259</div>
              </div>
              <div class="p-3 bg-zinc-800/80 rounded border border-zinc-700">
                <div class="text-zinc-400 text-[11px]">Cạn nhầm thành Nước</div>
                <div class="text-lg font-bold text-amber-400">251</div>
                <div class="text-[10px] text-zinc-400">Giảm từ 1.477 xuống 251</div>
              </div>
              <div class="p-3 bg-zinc-800/80 rounded border border-zinc-700">
                <div class="text-zinc-400 text-[11px]">Nước nhầm thành Cạn</div>
                <div class="text-lg font-bold text-amber-400">95</div>
                <div class="text-[10px] text-zinc-400">Giảm từ 275 xuống 95</div>
              </div>
              <div class="p-3 bg-zinc-800/80 rounded border border-zinc-700">
                <div class="text-zinc-400 text-[11px]">Đoán đúng Chim Nước</div>
                <div class="text-lg font-bold text-emerald-400">1.189</div>
              </div>
            </div>
            <div class="p-2.5 bg-emerald-950/20 border border-emerald-900/30 rounded text-xs text-emerald-200">
              ✅ <strong>Kết luận POTER:</strong> Cơ chế reweighting bằng thế kép Optimal Transport đã cân bằng phân phối biểu diễn, buộc mô hình nhận diện mỏ, vuốt, màu lông của chim thay vì nhìn màu nền.
            </div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <script>
    // Raw Data injected
    const APP_DATA = JSON_DATA_PLACEHOLDER;
    const metrics = APP_DATA.metrics;
    const errors = APP_DATA.errors;

    let currentTab = 'overview';
    let currentEpoch = 258;
    let isPlaying = false;
    let playInterval = null;

    let filteredErrors = [...errors];
    let currentPage = 1;
    const pageSize = 15;

    // Tab Switching
    function switchTab(tabId) {
      currentTab = tabId;
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.classList.add('text-[var(--muted-foreground,#a1a1aa)]');
      });
      document.getElementById('tab-btn-' + tabId).classList.add('active');
      document.getElementById('tab-btn-' + tabId).classList.remove('text-[var(--muted-foreground,#a1a1aa)]');

      ['overview', 'dynamics', 'causes', 'samples', 'matrix'].forEach(id => {
        const el = document.getElementById('tab-' + id);
        if (id === tabId) el.classList.remove('hidden');
        else el.classList.add('hidden');
      });

      if (tabId === 'dynamics') {
        renderCharts();
        updateEpochSnapshot(currentEpoch);
      } else if (tabId === 'samples') {
        renderErrorTable();
      }
    }

    // Epoch Scrubber logic
    function onEpochChange(val) {
      currentEpoch = parseInt(val);
      updateEpochSnapshot(currentEpoch);
    }

    function stepEpoch(delta) {
      currentEpoch = Math.max(0, Math.min(299, currentEpoch + delta));
      document.getElementById('epoch-slider').value = currentEpoch;
      updateEpochSnapshot(currentEpoch);
    }

    function jumpToEpoch(ep) {
      currentEpoch = ep;
      document.getElementById('epoch-slider').value = currentEpoch;
      updateEpochSnapshot(currentEpoch);
    }

    function togglePlay() {
      isPlaying = !isPlaying;
      const btn = document.getElementById('play-btn');
      if (isPlaying) {
        btn.innerText = '⏸ Dừng (Pause)';
        btn.classList.replace('bg-blue-600', 'bg-amber-600');
        playInterval = setInterval(() => {
          if (currentEpoch >= 299) {
            togglePlay();
            return;
          }
          stepEpoch(1);
        }, 150);
      } else {
        btn.innerText = '▶ Chạy (Play)';
        btn.classList.replace('bg-amber-600', 'bg-blue-600');
        clearInterval(playInterval);
      }
    }

    function updateEpochSnapshot(ep) {
      const d = metrics[ep];
      if (!d) return;

      document.getElementById('current-epoch-label').innerText = ep;
      document.getElementById('stat-val-wga').innerText = d.vw + '%';
      document.getElementById('stat-val-avg').innerText = d.va + '%';
      document.getElementById('stat-g1').innerText = d.g1 + '%';
      document.getElementById('stat-g2').innerText = d.g2 + '%';
      document.getElementById('stat-loss').innerText = d.tl + ' / ' + d.vl;

      let phase = 'Khởi động (Epoch 0-50)';
      let phaseColor = 'text-amber-400';
      if (ep === 258) {
        phase = 'Đỉnh Tối Ưu (Best WGA)';
        phaseColor = 'text-emerald-400';
      } else if (ep > 200) {
        phase = 'Bão Hòa & Đỉnh';
        phaseColor = 'text-cyan-400';
      } else if (ep > 50) {
        phase = 'Khử Tương Quan Giả';
        phaseColor = 'text-blue-400';
      }
      const phaseEl = document.getElementById('stat-phase');
      phaseEl.innerText = phase;
      phaseEl.className = 'text-xs font-semibold mt-1 ' + phaseColor;

      // redraw chart cursor
      drawWgaLossChart(ep);
      drawGroupsChart(ep);
    }

    // Chart Rendering on Canvas
    function renderCharts() {
      drawWgaLossChart(currentEpoch);
      drawGroupsChart(currentEpoch);
    }

    function drawWgaLossChart(activeEp) {
      const canvas = document.getElementById('chart-wga-loss');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      // Margin
      const pad = { top: 20, right: 20, bottom: 25, left: 35 };
      const cw = w - pad.left - pad.right;
      const ch = h - pad.top - pad.bottom;

      // Draw grid
      ctx.strokeStyle = '#27272a';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (ch / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();

        ctx.fillStyle = '#71717a';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText((100 - i * 25) + '%', pad.left - 5, y + 3);
      }

      // Plot Val WGA (0 to 100%)
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = 2;
      ctx.beginPath();
      metrics.forEach((m, idx) => {
        const x = pad.left + (idx / 299) * cw;
        const y = pad.top + (1 - m.vw / 100) * ch;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Plot Val Avg (0 to 100%)
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      metrics.forEach((m, idx) => {
        const x = pad.left + (idx / 299) * cw;
        const y = pad.top + (1 - m.va / 100) * ch;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Active Epoch Indicator Line
      const activeX = pad.left + (activeEp / 299) * cw;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(activeX, pad.top);
      ctx.lineTo(activeX, h - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Peak Indicator marker (Epoch 258)
      const peakX = pad.left + (258 / 299) * cw;
      const peakY = pad.top + (1 - 90.1 / 100) * ch;
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(peakX, peakY, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawGroupsChart(activeEp) {
      const canvas = document.getElementById('chart-groups');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const pad = { top: 20, right: 20, bottom: 25, left: 35 };
      const cw = w - pad.left - pad.right;
      const ch = h - pad.top - pad.bottom;

      // Grid
      ctx.strokeStyle = '#27272a';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (ch / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
        ctx.fillStyle = '#71717a';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText((100 - i * 25) + '%', pad.left - 5, y + 3);
      }

      const colors = { g0: '#60a5fa', g1: '#f87171', g2: '#fbbf24', g3: '#22d3ee' };

      // Plot each group
      ['g0', 'g1', 'g2', 'g3'].forEach(k => {
        ctx.strokeStyle = colors[k];
        ctx.lineWidth = k === 'g1' ? 2.5 : 1.5;
        ctx.beginPath();
        metrics.forEach((m, idx) => {
          const x = pad.left + (idx / 299) * cw;
          const y = pad.top + (1 - m[k] / 100) * ch;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });

      // Active cursor
      const activeX = pad.left + (activeEp / 299) * cw;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(activeX, pad.top);
      ctx.lineTo(activeX, h - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Filter and Pagination for Error Table
    function applyFilters() {
      const q = document.getElementById('search-input').value.toLowerCase();
      const g = document.getElementById('group-filter').value;
      const c = document.getElementById('conf-filter').value;
      const e = document.getElementById('erm-filter').value;

      filteredErrors = errors.filter(item => {
        // search
        if (q && !item.species.toLowerCase().includes(q) && !item.filename.toLowerCase().includes(q)) {
          return false;
        }
        // group
        if (g !== 'all' && item.group !== g) {
          return false;
        }
        // conf
        if (c === 'high' && item.conf < 0.85) return false;
        if (c === 'medium' && (item.conf < 0.65 || item.conf >= 0.85)) return false;
        if (c === 'low' && item.conf >= 0.65) return false;
        // erm
        if (e === 'both_wrong' && item.erm_pred === item.true) return false;
        if (e === 'erm_wrong_high' && (item.erm_pred === item.true || item.erm_conf < 0.80)) return false;

        return true;
      });

      currentPage = 1;
      renderErrorTable();
    }

    function changePage(delta) {
      const totalPages = Math.ceil(filteredErrors.length / pageSize) || 1;
      currentPage = Math.max(1, Math.min(totalPages, currentPage + delta));
      renderErrorTable();
    }

    function renderErrorTable() {
      const tbody = document.getElementById('error-table-body');
      if (!tbody) return;

      const total = filteredErrors.length;
      const totalPages = Math.ceil(total / pageSize) || 1;
      document.getElementById('filter-count').innerText = total;
      document.getElementById('current-page').innerText = currentPage;
      document.getElementById('total-pages').innerText = totalPages;
      document.getElementById('page-indicator').innerText = currentPage;

      document.getElementById('prev-page-btn').disabled = (currentPage === 1);
      document.getElementById('next-page-btn').disabled = (currentPage === totalPages);

      const start = (currentPage - 1) * pageSize;
      const pageItems = filteredErrors.slice(start, start + pageSize);

      if (pageItems.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="p-6 text-center text-zinc-500">Không tìm thấy ca lỗi nào phù hợp với bộ lọc.</td></tr>`;
        return;
      }

      let html = '';
      pageItems.forEach(item => {
        let badgeClass = 'badge-majority';
        if (item.group.includes('Landbird on Water')) badgeClass = 'badge-minority';
        else if (item.group.includes('Waterbird on Land')) badgeClass = 'badge-warning';

        const confPct = Math.round(item.conf * 100);
        let confColor = 'text-zinc-300';
        if (confPct >= 85) confColor = 'text-red-400 font-bold';
        else if (confPct < 65) confColor = 'text-amber-400';

        const ermMatch = (item.erm_pred !== item.true) ?
          `<span class="text-red-400 font-medium">Sai (${Math.round(item.erm_conf * 100)}%)</span>` :
          `<span class="text-emerald-400 font-medium">Đúng</span>`;

        html += `
          <tr class="hover:bg-zinc-800/40 transition-colors">
            <td class="p-3">
              <div class="font-semibold text-white">${item.species}</div>
              <div class="text-[11px] text-zinc-400 font-mono">${item.filename}</div>
            </td>
            <td class="p-3">
              <span class="badge ${badgeClass}">${item.group}</span>
            </td>
            <td class="p-3">
              <div class="flex items-center gap-1.5">
                <span class="capitalize text-zinc-300">${item.true}</span>
                <span class="text-zinc-500">&rarr;</span>
                <span class="capitalize text-red-400 font-semibold">${item.pred}</span>
              </div>
            </td>
            <td class="p-3">
              <span class="${confColor}">${(item.conf * 100).toFixed(1)}%</span>
            </td>
            <td class="p-3">
              ${ermMatch}
            </td>
            <td class="p-3 text-zinc-400 text-xs max-w-xs">
              ${item.reason}
            </td>
          </tr>
        `;
      });

      tbody.innerHTML = html;
    }

    // Initialize
    window.addEventListener('load', () => {
      updateEpochSnapshot(258);
      renderErrorTable();
    });

    window.addEventListener('resize', () => {
      if (currentTab === 'dynamics') renderCharts();
    });
  </script>
</body>
</html>
"""

final_html = html_content.replace("JSON_DATA_PLACEHOLDER", json_data_str)

output_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\47a3297c-bde9-49b5-ae5f-2760c7fc36d2\poter_waterbirds_error_explorer.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_html)

print("Generated HTML successfully at:", output_path)
print("File size:", os.path.getsize(output_path))
