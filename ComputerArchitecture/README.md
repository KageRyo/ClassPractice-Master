## 計算機結構作業二
614410073 張健勳  
1. 需要先確保環境有 gcc 和 python（並安裝 matplotlib）
2. 在 Windows 非 ARM 架構的 CPU 環境下執行
3. 目前的測量程式碼是針對  Intel Core i5-14500（509實驗室配的個人電腦）
4. 可以直接執行 run_benchmark_and_plot.ps1 腳本來完成
5. 結果會輸出成 csv 並再轉換成圖片

---

## 題目 2.5

If necessary, modify the code in Figure 2.32 to measure the following system characteristics.
Plot the experimental results with elapsed time on the y-axis and the memory stride on the x-axis.
Use logarithmic scales for both axes and draw a line for each cache size.

a. What is the system page size?

b. How many entries are there in the TLB?

c. What is the miss penalty for the TLB?

d. What is the associativity of the TLB?

## 中文答案與對應圖

a. 系統 page size 為 **4 KB**。  
對應圖：`tlb_page_size_probe.png`  
依據：在大工作集下，stride 到 4KB 附近出現明顯轉折。

b. **實驗觀測的有效條目數約為 512（effective entries）**，不是直接等同硬體規格值。  
對應圖：`tlb_entries_probe.png`  
依據：Pages touched 增加到約 512 時，延遲曲線出現主要膝點。  
補充：i5-14500 常見規格為 L1 DTLB 約 64、STLB 約 2048；本實驗值 512 反映的是此存取模式下的「有效覆蓋量」，可能混合多層 TLB 與 cache 影響。

c. TLB miss penalty 約為 **14.86 ns/access**（以 TLB 命中區與未命中區平均延遲差估計）。  
對應圖：`tlb_entries_probe.png`、`tlb_analysis_summary.txt`  
依據：TLB miss 區與 hit 區的平均延遲差。

d. **本實驗觀測到兩段轉折：第一段約 7（對應約 6-way），第二段約 13（對應約 12-way）**。  
對應圖：`tlb_associativity_conflict.png`、`tlb_assoc_benchmark.csv`  
說明：若用 N-1 規則，第一轉折給 6-way、第二轉折給 12-way。由於 i5-14500 已知 STLB 為 12-way，第二段與規格更一致；第一段可能是 L1/量測噪聲/混合效應。

## 圖表說明（如何對應題目）

1. `memory_lines_by_array_size.png`
- 用途：題目要求的主圖格式（y 軸 elapsed time / latency、x 軸 stride，雙對數）
- 讀法：每條線代表一個 array size（可視為不同 cache working-set 條件）

2. `tlb_page_size_probe.png`
- 對應題目：2.5(a) page size
- x 軸：Stride Bytes（log2）
- y 軸：Latency（log）
- 讀法：在大工作集下觀察明顯轉折，轉折落在 4KB 附近

3. `tlb_entries_probe.png`
- 對應題目：2.5(b) TLB entries、2.5(c) miss penalty
- x 軸：Pages touched（ArrayBytes / PageSize）
- y 軸：Latency（log）
- 讀法：主要膝點對應可覆蓋的頁數上限（entries）；膝點前後平台差對應 miss penalty

4. `tlb_associativity_conflict.png`
- 對應題目：2.5(d) associativity
- x 軸：Ways needed（強制同組 TLB set 的衝突頁數）
- y 軸：Latency
- 讀法：第一個穩定上升點在 way=7，依 N-1 規則推估約 6-way

5. `tlb_associativity_probe.png`
- 用途：輔助觀察圖（非 d 題主要證據）
- 讀法：從原始 memory benchmark 中只看 page-size 倍數 stride 的趨勢，容易混入多層 TLB/快取效應

6. `memory_heatmap.png` / `memory_textbook_style.png`
- 用途：整體行為展示與課本風格對照
- 讀法：幫助觀察 cache 與記憶體層級變化，但不是 TLB associativity 的主證據

## 補充說明（為了回答 d 題所做的修改）

原本的 memory mountain 圖主要擅長觀察 cache 與 page-stride 轉折，對 associativity 不夠直接。
因此在 `HW2_2.25.c` 新增了專用模式：

- `--tlb-assoc-only`：固定一組 page 間距（16p、32p、64p、128p），逐步增加同一組 TLB set 的衝突頁數（ways = 1..24）
- 輸出 `tlb_assoc_benchmark.csv`
- 再由 `plot_memory_benchmark.py` 繪出 `tlb_associativity_conflict.png`

## 執行方式

完整流程（原始 memory benchmark + 繪圖 + TLB 關聯圖）：

```powershell
./run_benchmark_and_plot.ps1
```

只重跑 d 題（TLB associativity 專用資料）：

```powershell
./HW2_2.25.exe --tlb-assoc-only
python plot_memory_benchmark.py memory_benchmark.csv
```