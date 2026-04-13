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

## 最終可繳交段落（可直接貼報告）

以下內容可直接作為本題最終回答。

a. 系統 page size 為 **4 KB**。對應圖為 `tlb_page_size_probe.png`。在大工作集下，stride 接近 4KB 時延遲出現穩定轉折，因此判定頁面大小為 4KB。

b. TLB entries 建議以「雙層解讀」回答。程式 heuristic 輸出（`tlb_analysis_summary.txt`）為 L1 約 **16 entries**、STLB 約 **4096 entries**；但考慮本方法會混入 cache 與 prefetch 影響，並與 i5-14500 常見規格交叉比對後，報告建議值為 L1 約 **64**、STLB 約 **2048**。因此本題答案應同時陳述「量測估計值」與「架構參考值」，並註明估計值屬於 heuristic。

c. TLB miss penalty 建議分兩段呈現。由 `tlb_analysis_summary.txt` 可得：L1 -> STLB penalty 約 **9.88 ns/access**，STLB miss penalty 約 **115.22 ns/access**。其中後者更接近實際 page walk 成本，因此本題主要答案建議採用 **約 115 ns/access**，並附註分段估計結果。

d. TLB associativity 以 dedicated conflict benchmark（`tlb_assoc_benchmark.csv`、`tlb_associativity_conflict.png`）判讀，轉折點出現在 way=7 與 way=13，依 N-1 規則對應約 **6-way** 與 **12-way**。因此可回答：L1 TLB 約 6-way，L2/STLB 約 12-way；此結果與常見 STLB 12-way 規格一致。

作答策略（哪幾題需要修正）：a 題可維持 4KB 不變；b、c、d 題需改為分層與限制說明寫法，避免把單次微基準結果直接視為硬體絕對規格。

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
- 讀法：找穩定跳升點，若在 way=N 發生，可用 N-1 估計 associativity；本輪資料噪聲較大，需搭配多次重跑判讀

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

## 多次重跑統計模板（建議放報告附錄）

建議至少重跑 3 次（Run1/Run2/Run3），把每次 `tlb_analysis_summary.txt` 的重點數值填入下表：

| 指標 | Run1 | Run2 | Run3 | 平均值 | 標準差 |
|---|---:|---:|---:|---:|---:|
| Page size (KB) |  |  |  |  |  |
| L1 TLB entries (pages) |  |  |  |  |  |
| STLB entries (pages) |  |  |  |  |  |
| L1 -> STLB penalty (ns/access) |  |  |  |  |  |
| STLB miss penalty (ns/access) |  |  |  |  |  |
| L1 associativity (way) |  |  |  |  |  |

報告建議文字（可直接引用）：

1. a 題（page size）在多次重跑中穩定為 4KB，判斷可信度高。
2. b、c 題受工作集與噪聲影響較大，建議同時報告「程式估計平均值」與「硬體規格參考值（L1 DTLB~64、STLB~2048）」。
3. d 題若各次轉折點分散，請註記 associativity 為 exploratory 結果，避免過度解讀單次 run。