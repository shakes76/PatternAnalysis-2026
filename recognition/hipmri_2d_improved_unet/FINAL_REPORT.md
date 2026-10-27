# HipMRI 2D U-Net 与 Improved U-Net 最终实验报告

Check-off已由学生于2026-10-26确认完成。

报告日期：2026年10月26日。报告汇总已完成的训练、验证、冻结测试及故障恢复，并单独列出最终模型的少量验证诊断。原最终实验汇总只读取保存证据并复算统计；后续验证案例的执行范围另有记录，未改变原训练或测试统计。

在资源敏感、以ID5分割为主要目标的原型场景，当前证据不足以支持默认升级到Improved。它的测试ID5 Dice仅高0.106个百分点，而参数量增加64.37%，训练阶段已记录的allocated峰值约为3.11倍；四组相同方法的合成前向profile中，median批延迟为baseline的3.43至7.98倍。更高recall和多类别收益可能有应用价值，但需要明确的误分／漏分代价与预算才能判断是否值得。以下工程分析保留测试前已冻结的模型与完整结果，不据测试分数重新选择epoch、阈值或参数。

## 实验目标和数据

任务是将单通道2D MRI切片分为6个原始类别ID，主要指标为ID5 Dice。原始提供方的ID映射已有核查；把课程2D派生数据的ID5解释为前列腺，仍采用用户接受的“导出保留原ID”工作假设，尚无独立无重标号证明。本实验没有更改mask编号。<a href="label_schema.json">[标签范围]</a>

<table>
<tr><th>划分</th><th>MRI／mask配对数</th><th>数值case代理组</th><th>用途</th></tr>
<tr><td>训练</td><td>11,460</td><td>31组</td><td>更新模型参数</td></tr>
<tr><td>验证</td><td>660</td><td>4组：36至39</td><td>选模、协议诊断与已有失败分析</td></tr>
<tr><td>测试</td><td>540</td><td>3组：40、41、42</td><td>权重及规则冻结后的最终比较</td></tr>
</table>

三个划分的数值case集合两两无交叉；同case的不同week或slice不能被当作不同患者拆开。case只是文件名中的身份代理，尚未独立确认患者映射。测试540对均为256×128，共17,694,720个原有效像素；NIfTI形状和Affine全部逐值匹配，但空间单位为unknown，不能据此证明解剖对齐或换算毫米。像素数不能当作同等数量的独立受试者。<a href="evidence/test_manifest_audit_20261026.json">[划分及头信息审计]</a>

## 模型和公平训练协议

Baseline是四次下采样的U-Net：双3×3卷积和ReLU、MaxPool、转置卷积上采样及skip拼接。Improved是本仓库依据既有核查记录中Isensee等（2018）残差U-Net架构思路实现的2D改编，加入预激活InstanceNorm／LeakyReLU、残差context、Dropout2d、stride卷积下采样、nearest上采样及多尺度logits逐级融合；仍由同一个CE监督，不另加辅助损失。这不是完整3D论文训练配方的复现，也没有消融实验来归因单个模块。两者base width同为16，但参数量没有匹配，因此比较的是这两个完整实现的效果与成本。<a href="modules.py">[实际模型代码]</a>

<table>
<tr><th>共同条件</th><th>固定值或规则</th></tr>
<tr><td>输入／输出</td><td>[B,1,H,W] MRI；[B,6,H,W]原始logits</td></tr>
<tr><td>归一化和补边</td><td>每片MRI min-max，常量片为0；右／下补至16倍数；同batch再补至最大尺寸</td></tr>
<tr><td>有效区域</td><td>所有原像素均有效，包括ID0和黑色MRI像素；人工补边用mask=-100并排除于loss和指标</td></tr>
<tr><td>优化</td><td>共享有效区域one-hot CE；Adam，lr=0.001，weight_decay=0；seed=3710</td></tr>
<tr><td>预算</td><td>每模型从头训练20完整轮；B4、num_workers=0；每轮2,865次更新，共57,300次</td></tr>
<tr><td>精度及其他设置</td><td>FP32；无增强、AMP、scheduler或early stopping；相同软件／GPU及backend数学设置</td></tr>
<tr><td>验证和测试</td><td>eval、关闭梯度、无optimizer更新；既有字典排序、B4、无shuffle、保留末批、六类argmax；无TTA或阈值调整</td></tr>
</table>

评估先累计整组有效像素混淆矩阵，再计算Dice／IoU；不是batch Dice平均，也不是case等权平均。前景macro排除ID0及预测和参考双空的未定义类；空参考仍预测该类时记0分。CE按有效像素数加权，不能从混淆矩阵重构。Improved的InstanceNorm会看到补边，因而批次顺序、组成及动态补边也属于共同流程。<a href="evidence/id5_run_plan_20261026.json">[固定协议]</a> <a href="evidence/frozen_test_recipe_20261026.json">[测试前冻结规则]</a>

记录环境为Python3.13.2、NumPy2.5.4、NiBabel5.4.2、PyTorch2.11.0+cu128、CUDA12.8、cuDNN91900及RTX4070 Ti，CPU threads=1。cuDNN deterministic=True、benchmark=False；matmul TF32=False、cuDNN TF32=True、float32_matmul_precision=highest。FP32描述模型／输入dtype且无AMP，不能另称所有backend均禁用TF32。<a href="evidence/id5_baseline_20261026.json">[运行环境记录]</a> <a href="evidence/id5_preflight_20261026.json">[依赖预检]</a>

## 验证选模和冻结

原实验按最低验证CE选中baseline第5轮、Improved第3轮。诊断发现Improved的最高ID5 Dice出现在后期，而CE最低轮并不对应ID5最高轮。用户随后在测试前批准两模型重新各训练20轮，统一按“最高验证ID5 Dice；精确并列时最低CE；再并列时最早epoch”选模，另存ID5-best、CE-best和last。原CE实验没有被覆盖。

<table>
<tr><th>验证阶段</th><th>Baseline epoch／ID5 Dice</th><th>Improved epoch／ID5 Dice</th></tr>
<tr><td>原CE选模实验</td><td>5／0.816567</td><td>3／0.800187</td></tr>
<tr><td>统一ID5选模实验</td><td>5／0.816567</td><td>17／0.834394</td></tr>
<tr><td>最终保留角色</td><td>ID5-best5、CE-best5、last20</td><td>ID5-best17、CE-best3、last20</td></tr>
</table>

统一规则下的验证ID5差为1.783个百分点，但验证前景macro Dice为0.903274与0.896437，Improved反而较低。两模型新旧20轮训练／验证数值及CE-best／last的模型与Adam状态实际逐值一致，故此处变化来自选模规则，不是新的训练算法增益；同seed重跑也不是独立seed重复实验。最高ID5轮次与最高概率拟合质量不能混为一个目标。<a href="evidence/formal_comparison_20261026.json">[原CE实验]</a> <a href="evidence/id5_comparison_20261026.json">[统一ID5实验]</a>

测试前冻结baseline第5轮和Improved第17轮的权重摘要、预处理及评估规则。两份冻结副本与原ID5-best字节一致，测试没有回头选其他轮次。第5／17轮是20轮候选中的选择，训练并未在这些轮次提前停止。

## 冻结测试结果

<table>
<tr><th>完整测试指标</th><th>Baseline第5轮</th><th>Improved第17轮</th><th>Improved减Baseline</th></tr>
<tr><td>ID5 Dice ↑</td><td>0.843161</td><td>0.844222</td><td>+0.001060，即+0.106个百分点</td></tr>
<tr><td>ID5 IoU ↑</td><td>0.728850</td><td>0.730435</td><td>+0.001586</td></tr>
<tr><td>前景macro Dice ↑</td><td>0.906227</td><td>0.915912</td><td>+0.009685</td></tr>
<tr><td>前景macro IoU ↑</td><td>0.835422</td><td>0.850269</td><td>+0.014847</td></tr>
<tr><td>有效像素CE ↓</td><td>0.055552</td><td>0.071634</td><td>+0.016081，损失较差</td></tr>
<tr><td>Pixel accuracy ↑</td><td>0.977617</td><td>0.978941</td><td>+0.001325</td></tr>
</table>

<table>
<tr><th> 原类别ID</th><th>Baseline Dice／IoU</th><th>Improved Dice／IoU</th></tr>
<tr><td>0</td><td>0.996669／0.993360</td><td>0.997206／0.994427</td></tr>
<tr><td>1</td><td>0.983693／0.967909</td><td>0.984636／0.969737</td></tr>
<tr><td>2</td><td>0.934470／0.877000</td><td>0.936415／0.880433</td></tr>
<tr><td>3</td><td>0.959706／0.922534</td><td>0.965891／0.934031</td></tr>
<tr><td>4</td><td>0.810103／0.680817</td><td>0.848396／0.736707</td></tr>
<tr><td>5</td><td>0.843161／0.728850</td><td>0.844222／0.730435</td></tr>
</table>

本批测试的ID5收益相对baseline约0.1257%，与“0.106个百分点”是不同口径；CE增加约28.95%，其损失差不能写成百分点。各类pooled Dice均略高，ID4增加约3.829个百分点，对macro改善贡献较大，但不代表每片或每case都改善。验证和测试来自不同case组，优势幅度缩小不能单独证明某个模块过拟合。

<table>
<tr><th>由测试混淆矩阵导出的ID5像素统计</th><th>Baseline</th><th>Improved</th></tr>
<tr><td>TP／FP／FN</td><td>48,408／6,382／11,627</td><td>53,099／12,660／6,936</td></tr>
<tr><td>Precision</td><td>0.883519</td><td>0.807479</td></tr>
<tr><td>Recall</td><td>0.806330</td><td>0.884467</td></tr>
</table>

两模型参考ID5均为60,035像素。Improved多检出4,691个真实ID5像素，同时多出6,278个FP，因此Dice只获得小幅净增。这里的precision／recall是保存矩阵的描述性展开，不是患者漏诊率，也未用于测试后调整阈值。两份完整评分均覆盖540片、135批、0更新，模型state前后相同。<a href="evidence/frozen_test_comparison_20261026.json">[完整测试及来源]</a>

![冻结测试的六类Dice，Improved为明确授权的完整补跑](evidence/frozen_test_comparison_20261026.png)

图1复用已保存聚合图，不包含MRI。图中分数分别来自原baseline完整遍历和获准的Improved完整补跑；中断、重复曝光及授权详见下节。

## 资源成本

<table>
<tr><th>项目</th><th>Baseline</th><th>Improved</th><th>Improved／Baseline</th></tr>
<tr><td>参数量</td><td>1,940,902</td><td>3,190,194</td><td>1.644倍，+64.37%</td></tr>
<tr><td>训练阶段最大allocated MiB</td><td>182.642</td><td>567.992</td><td>3.110倍，+210.99%</td></tr>
<tr><td>训练阶段最大reserved MiB</td><td>250.000</td><td>694.000</td><td>2.776倍</td></tr>
<tr><td>验证阶段最大allocated MiB</td><td>92.182</td><td>141.250</td><td>1.532倍</td></tr>
<tr><td>恢复阶段allocated MiB</td><td>52.998</td><td>84.844</td><td>1.601倍</td></tr>
<tr><td>20轮run墙钟秒</td><td>2,243.088</td><td>3,549.024</td><td>1.582倍，仅运行记录</td></tr>
</table>

上述内存来自每模型42个同步阶段记录，包含各阶段注明的模型、Adam、梯度／激活及分配器状态；reserved包含缓存。它们不是总GPU显存、增量显存或未测量的全进程峰值。本机两模型都已完成预算；不能仅凭比例声称Improved无法部署，也不能把训练墙钟比当作纯计算速度比。<a href="evidence/id5_comparison_20261026.json">[阶段资源证据]</a>

<table>
<tr><th>固定合成输入</th><th>Baseline median／p95 批毫秒</th><th>Improved median／p95 批毫秒</th><th>median倍数</th></tr>
<tr><td>B1，256×128</td><td>1.146／3.072</td><td>4.776／8.876</td><td>4.169</td></tr>
<tr><td>B1，256×144</td><td>1.432／1.574</td><td>11.024／12.495</td><td>7.697</td></tr>
<tr><td>B4，256×128</td><td>1.347／1.477</td><td>10.747／12.402</td><td>7.977</td></tr>
<tr><td>B4，256×144</td><td>3.134／4.028</td><td>10.756／12.267</td><td>3.432</td></tr>
</table>

Profile绑定已冻结第5／17轮：同一RTX4070 Ti、FP32及相同runtime，每组10次warmup、50次同步前向，p95为linear百分位；排除权重加载、I/O、传输、指标和绘图，无optimizer或测试像素。原始计时已用标准库独立重算median／p95。它是相同测量方法的本机单次快照，未隔离所有系统负载或提供重复实验置信区间。B4数值是整批时间，不能直接当单张延迟；3.43至7.98倍也不能推广到任意硬件。<a href="evidence/id5_baseline_profile_20261026.json">[Baseline计时]</a> <a href="evidence/id5_improved_profile_20261026.json">[Improved计时]</a>

完整测试阶段的3.611秒与8.732秒尤其不能作公平延迟对照：baseline使用原日志实现，Improved补跑使用fsync／重试修复，缓存与运行时状态也不同。本报告只保留这些实际记录，不用它们支撑架构速度结论。

## 已有验证失败案例

<strong>下列五例来自旧CE实验的baseline5、Improved3及last20，不是最终冻结Improved17的逐片失败分析，也不是测试失败案例。</strong>该历史诊断阶段没有推理第17轮；本轮获准补最终第5／17轮的少量验证展示，实际范围与结果另列，旧图仍是旧3／20轮诊断。案例按A至E的固定诊断规则依次去重选择，并非随机样本或失败率估计；所有分数按完整原图计算。<a href="evidence/selection_diagnosis_20261026.json">[案例规则、计数和图像索引]</a>

<table>
<tr><th>验证案例</th><th>参考ID5像素</th><th>旧baseline5 Dice</th><th>旧Improved3 Dice</th><th>旧Improved20 Dice</th><th>观察</th></tr>
<tr><td>A case038／week0／slice29</td><td>160</td><td>0</td><td>0</td><td>0.439024</td><td>两份best全漏；last20仅检出45像素，仍漏115</td></tr>
<tr><td>B case039／week5／slice41</td><td>0</td><td>0</td><td>0</td><td>0</td><td>FP分别269／236／231；参考为空时Dice不能表示检出成功</td></tr>
<tr><td>C case037／week0／slice9</td><td>19</td><td>0</td><td>0</td><td>0</td><td>三者均漏掉19像素；少像素标注需要单独留意</td></tr>
<tr><td>D case039／week7／slice25</td><td>184</td><td>0.698305</td><td>0.032086</td><td>0.866279</td><td>Improved后期TP从3至149，同时FP从0至11</td></tr>
<tr><td>E case037／week0／slice17</td><td>342</td><td>0.696532</td><td>0.840532</td><td>0.649457</td><td>后期FP从7至155、FN从89至103，局部退步</td></tr>
</table>

选图以旧Improved3为依据，baseline只作同片对照。R为参考ID5像素数，FN率指FN/R，A和C还要求FN＞0。A先找Improved3最低的参考阳性case/week组，再取参考像素≥20且FN率最高的片；B取空参考中FP最多；C取0＜参考像素＜20且FN率最高；D／E取第20轮相对第3轮Dice最大正／负变化，均只在参考像素≥20中选择。20像素仅为诊断分箱，不能称为临床“小前列腺”；空参考也不证明医学上结构不存在。原图和三预测并集外扩8像素的放大图均已保存，指标没有因裁图改变。

旧660张验证中，406张参考为空；至少有1个FP的片数为baseline5的17片、Improved3的8片、Improved20的23片。254张参考阳性中，完全漏掉ID5的片数分别24、40、15。后期漏分减少也可能伴随更多空参考误分，汇总分数和单个漂亮案例都不足以覆盖这些模式。提交前审查按任务书p13允许的README必要视觉材料，将五张既有图逐字节复制到evidence/submission_figures；没有重新读原始图像、推理或测试。图的旧验证模型角色不变。

## 测试中断和授权补跑

首次测试中baseline完整完成135批／540张。Improved第111次post-forward hook已得到444张输出，但写入guard时Path.replace抛WinError5；该批尚未返回共享循环计算指标，前110批累计统计只在已退出进程内存，无法从console四舍五入CE或剩余24批恢复整集分数。原attempt和guard保持incomplete，具体外部占用者／拒绝原因未证实，不能直接归因于OneDrive。

故障来自Codex新增持久化代码未处理单次Windows替换拒绝。只修复记录I/O：同目录唯一temp、flush／fsync、PermissionError有限6次替换重试及保留原异常，不重试模型forward。修复后121项检查与500次构造文件替换通过；补跑入口加入后完整129项检查通过。所有检查与故障记录均保留，不能因修复成功删掉原失败。

Codex说明前444张会重复并询问额外许可；用户明确答复“允许，仅补跑 Improved”。随后同一第17轮、同一数据与数学规则完成135批／540张，baseline没有再构建或前向。Improved累计为111＋135＝246次batch forward、984张次，前444张重复；最终分数只来自获准完整补跑，未混入或伪称恢复原部分统计。因此不能宣称两模型都只评估过一次。两份完整评分均0更新、无梯度，state前后相同，原失败和已完成baseline保持不变。<a href="evidence/frozen_test_failure_20261026.json">[原失败]</a> <a href="evidence/frozen_test_recovery_approval_20261026.json">[明确授权]</a> <a href="evidence/frozen_test_recovery_20261026.json">[补跑事实]</a>

## 工程判断和证据限制

如果目标主要是ID5重叠，且对显存、吞吐或维护成本敏感，这批数据没有显示足够的收益来支持普遍承担额外成本：ID5仅+0.106个百分点，参数+64.37%，已测训练allocated峰值3.11倍，现有前向计时也较慢。Baseline可作为低成本工程参照；这不改变本实验测试前冻结的Improved主模型或其评分，也不以测试结果倒选其他epoch。

如果应用特别重视减少ID5漏分或ID4等多类别效果，Improved的较高recall和macro可能有价值，但更多FP必须计入人工复核或后续流程成本。当前没有预先定义的错误代价、临床最小有意义差异或预算阈值，不能把像素改善直接兑换成临床收益或经济回报。可明确的结论是“额外收益存在，但对主要ID5目标较小，升级价值尚未证实”。

证据来自单seed、同一2D派生数据及3个测试case代理组；没有消融、独立患者映射、case等权统计、临床评审或物理距离指标。旧项目结果曾被接触，本报告不声称该测试数据在所有早期工作中从未见过。本轮没有据测试结果调整模型、训练或评估规则；不追加训练或再次测试来修饰结论。冻结模型的补充验证案例可在既有验证集按预定规则展示，并明确其后验、少量样例范围；若要再次选模或提出新的泛化结论，应另立设计与合适的独立数据，不能继续把已封存测试集当调参集。

## 证据索引和AI使用披露

关键数值由collect_final_report_evidence.py从19份已保存JSON重新提取，独立核对80组历史训练／验证混淆矩阵、完整验证／测试分数、四种profile输入组合的原始计时及授权来源链。该脚本只用标准库，不导入模型框架，不读取MRI、mask或权重，不训练或测试。<a href="evidence/final_report_summary_20261026.json">[本报告机器汇总]</a> <a href="collect_final_report_evidence.py">[只读汇总代码]</a>

<table>
<tr><th>内容</th><th>主要保存证据</th></tr>
<tr><td>训练条件、选模与资源</td><td>id5_run_plan_20261026.json；id5_comparison_20261026.json；两份id5_*_20261026.json及profile</td></tr>
<tr><td>原CE选模和已有失败</td><td>formal_comparison_20261026.json；selection_diagnosis_20261026.json</td></tr>
<tr><td>测试冻结和完整分数</td><td>frozen_test_recipe／plan／comparison_20261026.json；baseline与improved完整结果</td></tr>
<tr><td>中断、修复和额外授权</td><td>frozen_test_attempt／failure／io_fix／recovery_approval／recovery_20261026.json</td></tr>
</table>

表中JSON均位于evidence目录；完整文件名与SHA见机器汇总。正式英文报告见README。旧实现和测试结果暴露、Codex协助、真实失败及授权恢复继续按实际记录披露。数据与模型文件留在忽略目录，报告仅引用必要图片，不包含NIfTI或完整数据集。

AI使用披露：本仓库代码、自动运行、统计核对、故障修复及报告由Codex和辅助代理协助完成；用户提供已记录的概念回答、方案确认、运行与额外补跑授权。这些事实不等同于用户亲自运行、独立实现或已理解全部代码。学生仍对所交代码、公式、实验主张及准确引用负责。构造检查及验证展示以实际记录为准，不冒充旧实验当时已有。

## 参考文献

- Ronneberger, O., Fischer, P., &amp; Brox, T. (2015). <a href="https://arxiv.org/abs/1505.04597">U-Net: Convolutional Networks for Biomedical Image Segmentation</a>. arXiv:1505.04597。用于baseline拓扑思路；本项目same padding、base16、Adam不是原论文全部训练设置。
- Isensee, F., et al. (2018). <a href="https://arxiv.org/abs/1802.10508">Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge</a>. arXiv:1802.10508。课程Ref[2]；本项目是2D架构改编，非完整3D/Dice-loss/augmentation配方复现。
- CSIRO Data Access Portal. <a href="https://doi.org/10.25919/45t8-p065">Prostate MRI collection</a>（version 2，collection 52549）。原始类别说明及2026-10-25实际核查摘要见label_schema.json；课程派生保留ID仍为工作假设。本次网页工具未能重新访问支持文件，未把历史核查改写成本次新验证。
- COMP3710_Report_2026_v2.5，§2.2、§3–§8，PDF p5–6、12–17。课程本地任务书为项目要求及本次报告审查依据。

## 最终第5／17轮的五例验证补充展示

本节是训练与冻结测试之后完成的验证诊断补充，不是原测试或旧CE诊断当时已有的图。先固定历史A–E五个案例及顺序，再读取这5对验证MRI／mask，对冻结baseline第5轮与Improved第17轮各做5次B1前向，共10次；不创建优化器、不反向、不重选模型。选择规则及时间见<a href="evidence/final_validation_display_plan_20261026.json">预先计划</a>，实际源码摘要与完整逐类统计见<a href="evidence/final_validation_display_20261026.json">执行记录</a>。

两模型均eval／inference_mode，参数无梯度、state前后相同，冻结checkpoint SHA与原第5／17轮相同且前后未改。nib.load实际调用10次，涉及10个唯一验证文件；压缩文件摘要前后核查另会读其字节。没有读取真实test NIfTI／mask、调用完整测试／恢复脚本或产生新测试分数。独立GT与prediction直接来自本次原尺寸数组，不从旧PNG反推；数组仅保存在忽略目录供后续重绘。

<table>
<tr><th>案例case／week／slice</th><th>参考ID5像素</th><th>Baseline5 TP／FP／FN</th><th>Improved17 TP／FP／FN</th><th>Baseline5 Dice</th><th>Improved17 Dice</th></tr>
<tr><td>A 038／0／29</td><td>160</td><td>0／0／160</td><td>58／0／102</td><td>0.000000</td><td>0.532110</td></tr>
<tr><td>B 039／5／41</td><td>0</td><td>0／269／0</td><td>0／175／0</td><td>0.000000</td><td>0.000000</td></tr>
<tr><td>C 037／0／9</td><td>19</td><td>0／0／19</td><td>0／0／19</td><td>0.000000</td><td>0.000000</td></tr>
<tr><td>D 039／7／25</td><td>184</td><td>103／8／81</td><td>153／9／31</td><td>0.698305</td><td>0.884393</td></tr>
<tr><td>E 037／0／17</td><td>342</td><td>241／108／101</td><td>282／220／60</td><td>0.697540</td><td>0.668246</td></tr>
</table>

A中Improved检出58个ID5像素但仍漏102；B的空参考误分从269减为175，仍非正确空预测；C的19个参考像素两者均全漏；D的重叠改善；E的Improved虽减少FN，却因FP220而低于baseline。以上是预选案例的观察，不能当失败率、独立患者统计或新的整集比较。B1展示不替代正式B4评分：E中baseline本次FP108，旧B4诊断为109；checkpoint／输入字节不变，不能因此改写旧分数。批次形状及浮点执行差异未单独隔离，不把这一像素差归因于某个模块。

每图第一行为完整原图，第二行为同一个参考及两模型预测ID5并集外扩8像素的显示ROI；全部统计基于完整原尺寸、排除人工padding。列依次为MRI、独立六类GT、baseline独立prediction与ID5 TP／FN／FP叠图、Improved独立prediction与同类叠图。绿色TP、品红FN、青色FP是误差显示类别，不改mask原ID。ID5组织含义仍保留派生ID假设，case为文件名代理，单位unknown，base／apex位置未经确认。

### 最终验证案例A 038／0／29

![最终冻结模型baseline5／Improved17；validation；case_038_week_0_slice_29.nii.gz](evidence/final_validation_display_20261026/A_final_validation_epoch5_17.png)

### 最终验证案例B 039／5／41

![最终冻结模型baseline5／Improved17；validation；case_039_week_5_slice_41.nii.gz](evidence/final_validation_display_20261026/B_final_validation_epoch5_17.png)

### 最终验证案例C 037／0／9

![最终冻结模型baseline5／Improved17；validation；case_037_week_0_slice_9.nii.gz](evidence/final_validation_display_20261026/C_final_validation_epoch5_17.png)

### 最终验证案例D 039／7／25

![最终冻结模型baseline5／Improved17；validation；case_039_week_7_slice_25.nii.gz](evidence/final_validation_display_20261026/D_final_validation_epoch5_17.png)

### 最终验证案例E 037／0／17

![最终冻结模型baseline5／Improved17；validation；case_037_week_0_slice_17.nii.gz](evidence/final_validation_display_20261026/E_final_validation_epoch5_17.png)

本次test展示代码在构造CPU数据和临时checkpoint上完成11项定向检查，未在真实test样例执行；<a href="evidence/prediction_display_constructed_checks_20261026.json">构造检查</a>与上述真实验证推理分开记录。predict.py是实验完成后的展示入口扩展，原实验源码与展示入口分别核验；<a href="evidence/submission_source_evolution_20261026.json">源码演进记录</a>将旧预测器与新增helper和当前入口分别核验，没有改旧recipe、分数或补跑事实。
