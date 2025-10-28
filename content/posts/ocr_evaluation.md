---
title: "为什么你的 DeepSeek-OCR 不好用？DeepSeek 与 PaddleOCR-VL 全面测评"
date: 2025-10-28T10:39:10+08:00
draft: false
tags: ["技术","VLM","OCR"]
categories: ["VLM"]
---

# 引言
OCR 不再只是识字。随着多模态大模型的发展，新一代的 OCR 模型（如 DeepSeek-OCR）已经具备了对图文内容进行理解、总结、翻译的能力。而传统 OCR 模型（如 PaddleOCR-VL）在识别精度与稳定性上依然有着无可替代的优势。本文将通过多个真实场景，横向评测这两种模型的差异与适用性。

# 模型简介
## PaddleOCR-VL

来自百度 PaddleOCR 团队，属于传统 OCR 扩展，支持文档版式分析（layout）、表格结构提取等。

### 功能简介
```python
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png

# 通过 --use_doc_orientation_classify 指定是否使用文档方向分类模型
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_doc_orientation_classify True

# 通过 --use_doc_unwarping 指定是否使用文本图像矫正模块
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_doc_unwarping True

# 通过 --use_layout_detection 指定是否使用版面区域检测排序模块
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_layout_detection False
```
更多功能使用参见官方文档 https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html


## DeepSeek-OCR

属于视觉语言模型（VLM）范畴，不仅识别，还能基于 prompt 输出总结、翻译、问答等。

### 功能简介
DeepSeek-OCR 使用不同的 prompt 来实现不同的功能
```python
# 将文档转换成 markdown 格式。
document: <image>\n<|grounding|>Convert the document to markdown. 
# 只提取文字，不需要任何格式。
other image: <image>\n<|grounding|>OCR this image. 
# 按照原文件内容格式的解析，以 markdown 格式进行输出。
without layouts: <image>\nFree OCR. 
 # 按照合理的内容格式解析输出，可能是 markdown 也可能是 html 或其他。
figures in document: <image>\nParse the figure.
# 理解并描述这张图片。
general: <image>\nDescribe this image in detail. 
```
更多功能使用参见官方文档 https://github.com/deepseek-ai/DeepSeek-OCR/tree/main

> 在实际测试过程中发现，输出结果不能完全能遵循 prompt 的设计，比如同一个表格，使用 `Free OCR` 和 `Parse the figure` 识别出的信息准确率不同（可参见下文复杂表格一栏），需要用户多实践总结最优方案。

两个模型实现原理上更多对比可参考博客《一图胜千言：DeepSeek、Glyph 与 PaddleOCR-VL 的不同答案》。

# 测评场景

## 常见 PDF 文档
![示例](../../static/images/pdf_com.jpg)
#### PaddleOCR-VL
![BD识别结果](../../static/images/pdf_com_bd.jpg)
文档提取
```
### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate  $ P_{drop} = 0.1 $ , instead of 0.3.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty  $ \alpha = 0.6 $  [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length +50, but terminate early when possible [38].

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU $ ^{5} $ .
```
#### DeepSeek-OCR
![DS识别结果](../../static/images/pdf_com_ds.jpg)

文档提取
```
6.1MachineTranslation[[36, 29, 287, 64]]
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)[[35, 111, 937, 155]]
in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0[[35, 155, 937, 199]]
BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is[[35, 199, 937, 241]]
listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model[[35, 243, 937, 285]]
surpasses all previously published models and ensembles, at a fraction of the training cost of any of[[35, 287, 937, 329]]
the competitive models.[[35, 331, 252, 370]]
On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,[[35, 390, 938, 432]]
outperforming all of the previously published single models, at less than 1/4 the training cost of the[[35, 434, 937, 476]]
previous state-of-the-art model. The Transformer (big) model trained for English-to-French used[[35, 478, 937, 520]]
dropout rate Pdrop = 0.1, instead of 0.3.[[35, 520, 404, 562]]
For the base models, we used a single model obtained by averaging the last 5 checkpoints, which[[35, 580, 937, 622]]
were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We[[35, 624, 937, 666]]
used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters[[35, 668, 937, 710]]
were chosen after experimentation on the development set. We set the maximum output length during[[35, 712, 937, 754]]
inference to input length + 50, but terminate early when possible [38].[[35, 754, 671, 796]]
Table 2 summarizes our results and compares our translation quality and training costs to other model[[35, 816, 937, 858]]
architectures from the literature. We estimate the number of floating point operations used to train a[[35, 860, 937, 902]]
model by multiplying the training time, the number of GPUs used, and an estimate of the sustained[[35, 904, 937, 946]]
single-precision floating-point capacity of each GPU 5.[[35, 944, 535, 986]]
```
**显然 DeepSeek-OCR 没有很好的识别处理行内公式。**
## 模糊 PDF 文档
![示例](../../static/images/pdf_dim.jpg)

#### PaddleOCR-VL
![BD识别结果](../../static/images/pdf_dim_bd.jpg)
```
图书在版编目(CIP)数据

外国短篇小说经典100篇/欧·亨利等著；王永年等译．-北京：人民文学出版社，2003.7

(外国文学经典百篇系列)

ISBN 7 02-004195-7

Ⅰ．外… Ⅱ．①欧…②王… Ⅲ．短篇小说－作品集 世界 近代 Ⅳ．114

中国版本图书馆 CIP 数据核字(2003)第 014690 号

责任编辑：刘开华 责任校对：王鸿宝

装帧设计：柳泉 责任印制：李博

# 外国短篇小说经典 100 篇

# Wai Guo Duan Pian Xiao Shuo Jing Dian Yi Bai Pian

人民文学出版社出版

http://www.rw-cn.com

北京市朝内大街166号 邮编：100705

北京冠中印刷厂印刷 新华书店经销

字数 602 千字 开本  $ 890 \times 1240 $  毫米 1/32 印张 22.75 插页 2

2003 年 7 月 北京 第 1 版

2003年7月第1次印刷

印数：1—8000

ISBN 7 02 004195 7/1·3183

定价：32.00元
```
#### DeepSeek-OCR
![DS识别结果](../../static/images/pdf_dim_ds.jpg)
```
（京）新登字002号[[33, 50, 226, 68]]
图书在版编目（CIP）数据[[202, 220, 425, 238]]
外国短篇小说经典100篇/欧·亨利等著；王永年等译.－北[[202, 252, 730, 270]]
京：人民文学出版社，2003.7[[184, 278, 433, 295]]
（外国文学经典白篇系列）[[202, 303, 433, 320]]
ISBN702-004195-7[[202, 327, 435, 344]]
I.外…Ⅱ.①欧②EⅢ.短篇小说-作品[[205, 359, 662, 377]]
集世界近代IV.114[[184, 384, 411, 402]]
中国版本图书馆CIP数据核字（2003）第014690号[[202, 416, 653, 434]]
责任编辑：刘开华责任校对：王鸿宝[[264, 468, 649, 486]]
装帧设计：柳泉责任印制：李博[[264, 495, 649, 513]]
外国短篇小说经典100篇[[320, 607, 588, 625]]
Wai Guo Duan Pian Xiao Shuo Jing Dian Yi Bai Pian[[180, 634, 730, 654]]
人民文学出版社出版[[320, 685, 590, 702]]
http://www.rw-cn.com[[345, 710, 563, 728]]
北京市朝内大街166号邮编：100705[[320, 733, 590, 750]]
北京冠中印刷厂印刷新华书店经销[[303, 755, 601, 772]]
字数602千字开本890×1240毫米1/32印张22.75插页2[[225, 777, 679, 794]]
2003年7月北京第1版[[225, 799, 393, 813]]
2003年7月第1次印刷[[510, 799, 679, 816]]
印数：1-8000[[401, 819, 500, 834]]
ISN7[[338, 841, 388, 856]]
02[[384, 842, 420, 854]]
0041957/1·3183[[414, 841, 565, 856]]
定价：32.00元[[399, 861, 504, 879]]
```
**对于较为模糊的 pdf 扫描文档，两个模型都能够很好的识别提取**

## 复杂表格
![示例](../../static/images/pdf_table.jpg)

#### PaddleOCR-VL
![BD识别结果](../../static/images/pdf_table_bd.jpg)
文档提取
```
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.​

<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Model</td><td colspan="2">BLEU</td><td colspan="2">Training Cost (FLOPs)</td></tr><tr><td style='text-align: center;'>EN-DE</td><td style='text-align: center;'>EN-FR</td><td style='text-align: center;'>EN-DE</td><td style='text-align: center;'>EN-FR</td></tr><tr><td style='text-align: center;'>ByteNet [18]</td><td style='text-align: center;'>23.75</td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Deep-Att + PosUnk [39]</td><td style='text-align: center;'></td><td style='text-align: center;'>39.2</td><td style='text-align: center;'></td><td style='text-align: center;'>1.0  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>GNMT + RL [38]</td><td style='text-align: center;'>24.6</td><td style='text-align: center;'>39.92</td><td style='text-align: center;'>2.3  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'>1.4  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>ConvS2S [9]</td><td style='text-align: center;'>25.16</td><td style='text-align: center;'>40.46</td><td style='text-align: center;'>9.6  $ \cdot $  10 $ ^{18} $</td><td style='text-align: center;'>1.5  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>MoE [32]</td><td style='text-align: center;'>26.03</td><td style='text-align: center;'>40.56</td><td style='text-align: center;'>2.0  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'>1.2  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>Deep-Att + PosUnk Ensemble [39]</td><td style='text-align: center;'></td><td style='text-align: center;'>40.4</td><td style='text-align: center;'></td><td style='text-align: center;'>8.0  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>GNMT + RL Ensemble [38]</td><td style='text-align: center;'>26.30</td><td style='text-align: center;'>41.16</td><td style='text-align: center;'>1.8  $ \cdot $  10 $ ^{20} $</td><td style='text-align: center;'>1.1  $ \cdot $  10 $ ^{21} $</td></tr><tr><td style='text-align: center;'>ConvS2S Ensemble [9]</td><td style='text-align: center;'>26.36</td><td style='text-align: center;'>41.29</td><td style='text-align: center;'>7.7  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'>1.2  $ \cdot $  10 $ ^{21} $</td></tr><tr><td style='text-align: center;'>Transformer (base model)</td><td style='text-align: center;'>27.3</td><td style='text-align: center;'>38.1</td><td style='text-align: center;'>3.3  $ \cdot $  10 $ ^{18} $</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Transformer (big)</td><td style='text-align: center;'>28.4</td><td style='text-align: center;'>41.8</td><td style='text-align: center;'>2.3  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'></td></tr></table>
```
表格以 html 源文件格式返回，转化后如下：
<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Model</td><td colspan="2">BLEU</td><td colspan="2">Training Cost (FLOPs)</td></tr><tr><td style='text-align: center;'>EN-DE</td><td style='text-align: center;'>EN-FR</td><td style='text-align: center;'>EN-DE</td><td style='text-align: center;'>EN-FR</td></tr><tr><td style='text-align: center;'>ByteNet [18]</td><td style='text-align: center;'>23.75</td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Deep-Att + PosUnk [39]</td><td style='text-align: center;'></td><td style='text-align: center;'>39.2</td><td style='text-align: center;'></td><td style='text-align: center;'>1.0  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>GNMT + RL [38]</td><td style='text-align: center;'>24.6</td><td style='text-align: center;'>39.92</td><td style='text-align: center;'>2.3  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'>1.4  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>ConvS2S [9]</td><td style='text-align: center;'>25.16</td><td style='text-align: center;'>40.46</td><td style='text-align: center;'>9.6  $ \cdot $  10 $ ^{18} $</td><td style='text-align: center;'>1.5  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>MoE [32]</td><td style='text-align: center;'>26.03</td><td style='text-align: center;'>40.56</td><td style='text-align: center;'>2.0  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'>1.2  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>Deep-Att + PosUnk Ensemble [39]</td><td style='text-align: center;'></td><td style='text-align: center;'>40.4</td><td style='text-align: center;'></td><td style='text-align: center;'>8.0  $ \cdot $  10 $ ^{20} $</td></tr><tr><td style='text-align: center;'>GNMT + RL Ensemble [38]</td><td style='text-align: center;'>26.30</td><td style='text-align: center;'>41.16</td><td style='text-align: center;'>1.8  $ \cdot $  10 $ ^{20} $</td><td style='text-align: center;'>1.1  $ \cdot $  10 $ ^{21} $</td></tr><tr><td style='text-align: center;'>ConvS2S Ensemble [9]</td><td style='text-align: center;'>26.36</td><td style='text-align: center;'>41.29</td><td style='text-align: center;'>7.7  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'>1.2  $ \cdot $  10 $ ^{21} $</td></tr><tr><td style='text-align: center;'>Transformer (base model)</td><td style='text-align: center;'>27.3</td><td style='text-align: center;'>38.1</td><td style='text-align: center;'>3.3  $ \cdot $  10 $ ^{18} $</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Transformer (big)</td><td style='text-align: center;'>28.4</td><td style='text-align: center;'>41.8</td><td style='text-align: center;'>2.3  $ \cdot $  10 $ ^{19} $</td><td style='text-align: center;'></td></tr></table>


#### DeepSeek-OCR
使用两种不同的 prompt，得到了两种结果。

**Free OCR**
```
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

| Model                           | BLEU  | Training Cost (FLOPs) |
| ------------------------------- | ----- | --------------------- |
|                                 | EN-DE | EN-FR                 |
| ByteNet [18]                    | 23.75 |                       |
| Deep-Att + PosUnk [39]          | 39.2  | 1.4 · 10^20           |
| GNMT + RL [38]                  | 24.6  | 2.3 · 10^19           |
| ConvS2S [9]                     | 25.16 | 9.6 · 10^18           |
| MoE [32]                        | 26.03 | 2.0 · 10^19           |
| Deep-Att + PosUnk Ensemble [39] | 40.4  | 8.0 · 10^20           |
| GNMT + RL Ensemble [38]         | 26.30 | 1.8 · 10^20           |
| ConvS2S Ensemble [9]            | 26.36 | 7.7 · 10^19           |
| Transformer (base model)        | 27.3  | 3.3 · 10^18           |
| Transformer (big)               | 28.4  | 2.3 · 10^19           |
| """                             |       |                       |
```
**Parse the figure**
```
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-Frenchnewstest2014 tests at a fraction of the training cost.

| Model                           | BLEU  | Training Cost (FLOPs) |
| ------------------------------- | ----- | --------------------- |
| **EN-DE**                       |       |                       |
| ByteNet [18]                    | 23.75 |                       |
| Deep-Att + PosUnk [39]          | 39.2  | 1.0 * 10^20           |
| GNMT + RL [38]                  | 24.6  | 2.3 * 10^19           |
| ConvS2S [9]                     | 25.16 | 9.6 * 10^18           |
| MoE [32]                        | 26.03 | 2.0 * 10^19           |
| **EN-FR**                       |       |                       |
| ByteNet [18]                    | 23.75 |                       |
| Deep-Att + PosUnk Ensemble [39] | 39.2  | 1.0 * 10^20           |
| GNMT + RL Ensemble [38]         | 24.6  | 1.4 * 10^20           |
| ConvS2S [9]                     | 25.16 | 1.5 * 10^20           |
| MoE [32]                        | 26.03 | 1.2 * 10^20           |
| **EN-FR**                       |       |                       |
| Deep-Att + PosUnk Ensemble [39] | 40.4  | 8.0 * 10^20           |
| GNMT + RL Ensemble [38]         | 26.30 | 1.1 * 10^21           |
| ConvS2S Ensemble [9]            | 26.36 | 1.2 * 10^21           |
| **EN-FR**                       |       |                       |
| Transformer (base model)        | 27.3  | 3.3 * 10^18           |
| Transformer (big)               | 28.4  | 2.3 * 10^19           |
```
**PaddleOCR-VL 很好的识别并还原了表格信息；DeepSeek-OCR 使用不同 prompt 都能完成表格内容的识别，但都不够准确，使用`Free OCR`识别出的结果缺少了 BLEU 下 EN-FR 和 Training Cost (FLOPs) 下 EN-DE 的信息，`Parse the figure`识别的结果将 EN-DE 识别成了 EN-FR**
## 模糊表格
![示例](../../static/images/table.jpg)
#### DeepSeek-OCR
![DS识别结果](../../static/images/table_ds.jpg)
文档提取
```
出填关别[[328, 161, 384, 187]]
出口日期[[489, 164, 544, 190]]
申报日期[[661, 168, 715, 194]]
备案号[[795, 170, 838, 197]]
境外收货人[[32, 206, 99, 232]]
运输方式[[327, 213, 384, 239]]
运输工具名称及航次号[[489, 216, 620, 245]]
提运单号[[660, 220, 715, 249]]
监管方式[[327, 264, 384, 290]]
征免性质[[489, 270, 544, 296]]
许可证号[[660, 273, 715, 300]]
一般贸易[[328, 293, 380, 319]]
一般征税[[490, 296, 541, 322]]
合同协议号[[32, 315, 99, 340]]
贸易国（地区）[[327, 320, 416, 345]]
运抵国（地区）[[489, 325, 576, 350]]
指运港[[660, 330, 703, 356]]
俄罗斯[[327, 345, 370, 372]]
俄罗斯[[489, 350, 530, 376]]
符拉迪沃斯托克/海参崴（俄罗斯）[[660, 356, 848, 382]]
包装种类[[31, 368, 85, 393]]
纸箱[[31, 393, 60, 416]]
件数[[327, 376, 357, 404]]
毛重[[400, 378, 427, 402]]
净重（千克）[[489, 381, 565, 406]]
FOB[[566, 409, 592, 432]]
成交方式[[570, 381, 630, 406]]
运费[[660, 384, 689, 412]]
保费[[760, 388, 789, 416]]
杂费[[864, 390, 893, 418]]
标记唛码及备注[[29, 473, 123, 499]]
项号商品编号[[28, 555, 121, 580]]
商品名称及规格型号[[145, 558, 256, 583]]
数量及单位[[370, 560, 435, 586]]
单价/总价/币制[[475, 562, 565, 587]]
原产国（地区）[[592, 565, 680, 590]]
最终目的国（地区）境内货源地[[699, 567, 890, 596]]
征免[[931, 576, 961, 600]]
1[[30, 590, 40, 604]]
84818090.00[[51, 586, 120, 607]]
水龙头[[165, 586, 205, 612]]
套[[420, 592, 435, 612]]
1[[475, 592, 490, 612]]
中国[[640, 595, 667, 619]]
俄罗斯[[735, 595, 786, 621]]
（用途：厨房浴室用，品牌：ROSSINKA品牌，规格：1/2"，单位：套）[[106, 610, 468, 645]]
2[[30, 640, 39, 657]]
84819090.00[[52, 640, 120, 661]]
水龙头配件[[185, 640, 241, 661]]
个[[380, 643, 420, 669]]
中国[[602, 650, 630, 674]]
俄罗斯[[710, 650, 750, 676]]
（用途：厨房浴室用，品牌：ROSSINKA品牌，规格：1/2"，单位：个）[[111, 664, 468, 699]]
3[[28, 692, 39, 710]]
4[[28, 718, 39, 737]]
5[[28, 744, 39, 763]]
特殊关系确认：否[[156, 777, 258, 806]]
价格影响确认：否[[333, 781, 430, 810]]
支付特许权使用费确认：否[[525, 786, 671, 816]]
自报自缴：[[765, 792, 822, 818]]
申报人员[[26, 808, 81, 834]]
申报人员证号[[133, 808, 210, 834]]
电话[[301, 812, 331, 836]]
兹申明以上内容承担如实申报、依法纳税之法律责任[[457, 816, 740, 846]]
海关批注及签章[[760, 824, 853, 849]]
申报单位[[26, 857, 81, 883]]
中报单位（盖章：[[550, 870, 647, 895]]
```
**PaddleOCR-VL对此直接罢工报错，DeepSeek-OCR使用 OCR Free 也未能成功提取到表格格式和文字，只能使用存文字提取模式和描述模式。**
## 数学公式1
![示例](../../static/images/math.png)
#### PaddleOCR-VL
![BD识别结果](../../static/images/math_bd.png)
文档提取
```
题目：设有一个随机变量 X 服从均值为  $ \mu $  和方差为  $ \sigma^{2} $  的正态分布  $ N(\mu, \sigma^{2}) $ 。考虑二维随机变量  $ (X, Y) $ ，其中  $ Y = aX + b $ ， $ a \neq 0 $ ，b 为常数。请完成以下任务：

1. 计算 Y 的分布及其均值和方差。

2. 设  $ Z = X + Y $ ，求 Z 的均值和方差。

3. 若 X 和 Y 的协方差为  $ \mathrm{Cov}(X,Y)=\sigma_{XY} $ ，求 Z 和 X 的相关系数  $ \rho(Z,X) $ 。

解答：

1. 首先，Y 的分布为线性变换后的正态分布，故 Y 也服从正态分布。其均值和方差为：

 $$ E[Y]=E[aX+b]=aE[X]+b=a\mu+b $$ 

 $$ \operatorname{V a r}(Y)=\operatorname{V a r}(a X+b)=a^{2}\operatorname{V a r}(X)=a^{2}\sigma^{2} $$ 

因此， $ Y \sim N(a\mu + b, a^{2}\sigma^{2}) $ 。

2. 对于  $ Z = X + Y $ ，我们可以计算其均值和方差：

 $$ E[Z]=E[X+Y]=E[X]+E[Y]=\mu+(a\mu+b)=(1+a)\mu+b $$ 

 $$ \operatorname{Var}(Z) = \operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y) = \sigma^2 + a^2\sigma^2 + 2\sigma_{XY} $$ 

综上，Z 的均值为  $ (1+a)\mu+b $ ，方差为  $ \sigma^{2}+a^{2}\sigma^{2}+2\sigma_{XY} $ 。

3. 最后，相关系数  $ \rho(Z, X) $  的计算如下：

 $$ \rho(Z,X)=\frac{\operatorname{Cov}(Z,X)}{\sqrt{\operatorname{Var}(Z)\cdot\operatorname{Var}(X)}} $$ 

其中

 $$ \operatorname{Cov}(Z,X)=\operatorname{Cov}(X+Y,X)=\operatorname{Cov}(X,X)+\operatorname{Cov}(Y,X)=\operatorname{Var}(X)+\operatorname{Cov}(Y,X)=\sigma^{2}+\sigma_{XY} $$ 

代入后，得到：

 $$ \rho(Z,X)=\frac{\sigma^{2}+\sigma_{XY}}{\sqrt{(\sigma^{2}+a^{2}\sigma^{2}+2\sigma_{XY})(\sigma^{2})}} $$ 

以上是本题的解答。
```
#### DeepSeek-OCR
![DS识别结果](../../static/images/math_ds.jpg)
文档提取
```
题目：设有一个随机变量X服从均值为μ和方差为o²的正态分布N(μ,o²)。考虑二维随机变量（X，Y)，[[164, 60, 822, 90]]
其中Y=aX+b，a≠0，b为常数。请完成以下任务：[[164, 96, 512, 125]]
1.计算Y的分布及其均值和方差。[[183, 156, 394, 181]]
2.设Z=X+Y，求Z的均值和方差。[[183, 192, 424, 217]]
3.若X和Y的协方差为Cov(X,Y)=oy，求Z和X的相关系数p(Z,X)。[[183, 228, 660, 253]]
解答：[[164, 284, 201, 312]]
1.首先，Y的分布为线性变换后的正态分布，故Y也服从正态分布。其均值和方差为：[[183, 344, 725, 369]]
E[Y] = E[aX + b] = aE[X] + b = aμ + b[[394, 379, 640, 404]]
Var(Y) = Var(aX + b) = a²Var(X) = a²o²[[388, 409, 647, 439]]
因此，Y~N(aμ + b,a²o²)。[[199, 447, 371, 472]]
2.对于Z=X+Y，我们可以计算其均值和方差：[[183, 483, 489, 508]]
E[Z] = E[X + Y] = E[X] + E[Y] = μ + (aμ + b) = (1 + a)μ + b[[327, 517, 707, 547]]
Var(Z) = Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y) = o² + a²o² + 2oy[[279, 550, 749, 580]]
综上，Z的均值为（1+a)μ+b，方差为o²+a²o²+2oy。[[199, 586, 570, 611]]
3.最后，相关系数p(Z,X)的计算如下：[[183, 621, 430, 646]]
Cov(Z,X)[[512, 655, 580, 680]]
p(Z,X) =[[402, 667, 469, 697]]
√Var(Z)·Var(X)[[470, 685, 605, 715]]
其中[[199, 721, 229, 750]]
Cov(Z,X) = Cov(X + Y,X) = Cov(X,X) + Cov(Y,X) = Var(X) + Cov(Y,X) = o² + oxy[[235, 755, 790, 785]]
代入后，得到：[[199, 790, 290, 816]]
p(Z,X) =[[402, 830, 470, 860]]
o²+oy[[524, 820, 584, 846]]
√(o²+a²o²+2oy)(o²)[[469, 860, 626, 890]]
以上是本题的解答。[[165, 921, 286, 947]]
```
**显然 DeepSeek-OCR 对于专业数学符号的提取处理远不如 PaddleOCR-VL，例如：将 rho 识别成了 p，将 sigma 识别成了 o。**

## 数学公式2
![示例](../../static/images/pdf_math.jpg)
#### PaddleOCR-VL
![BD识别结果](../../static/images/pdf_math_bd.jpg)
文档提取
```
### 5.3 Optimizer

We used the Adam optimizer [20] with  $ \beta_{1}=0.9 $ ,  $ \beta_{2}=0.98 $  and  $ \epsilon=10^{-9} $ . We varied the learning rate over the course of training, according to the formula:

 $$ l rate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5}) $$ 

This corresponds to increasing the learning rate linearly for the first  $ warmup\_steps $  training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used  $ warmup\_steps = 4000 $ .
```
#### DeepSeek-OCR
![DS识别结果](../../static/images/pdf_math_ds.jpg)
文档提取
```
5.3Optimizer[[30, 52, 186, 123]]
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10-9. We varied the learning[[30, 210, 953, 290]]
rate over the course of training, according to the formula:[[30, 290, 562, 365]]
(3)[[925, 500, 958, 590]]
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,[[30, 660, 955, 740]]
and decreasing it thereafter proportionally to the inverse square root of the step number. We used[[30, 740, 955, 816]]
warmup_steps = 4000.[[30, 816, 264, 895]]
```
**DeepSeek-OCR 的识别直接无视了公式**
## 手写体1
![示例](../../static/images/handwriter.jpg)
#### PaddleOCR-VL
![BD识别结果](../../static/images/handwriter_bd.jpg)
文档提取
```
公安部正在修改户籍改革文件报国务院审批，目前已启动对户口法的立法调研。这是公安部办公厅副主任、新闻发言人武和平19日在公安部召开的新闻发布会上透露的。

武和平说，早在上世纪九十年代，我国就提出了取消农业、非农业的二元户口性质，来统一城乡户口登记制度，实行居住地户口登记制度的改革目标。这一目标提出后，经过国家10多个部门的共同努力，在国务院的统一领导下，城乡一体、以户口登记制度为重点的户籍制度改革积极稳妥地推进。去年，全国共有12个省、自治区、直辖市先后取消了二元户口划分。
```
#### DeepSeek-OCR
文档提取
```
公安部正在修改户籍改革文件报国务院审批，目前已启动对户[[65, 12, 965, 90]]
口法的立法调研。这是公安部办公厅副主任、新闻发言人武和平19日[[20, 115, 975, 195]]
在公安部召开的新闻发布公上透露的。[[18, 220, 560, 303]]
武和平说，早在上世纪九十年代，我国就提出了取消农业，非农业[[49, 325, 981, 410]]
的二元户口性质，来统一城乡户口登记制度，实行层住地户口制度登记[[16, 432, 987, 520]]
制度的改革目标。这一目标提出后，经过国家10多个部门的共同努力，[[12, 540, 987, 627]]
在国务院的统一领导下，城乡一体、以户口登记制度为重点的户籍[[12, 643, 963, 735]]
制度改革积极稳妥地推进。去年，全国共有12个省、自治区、[[12, 752, 951, 852]]
直辖市先后取消了二元户口划分。[[12, 875, 480, 965]]
```
**对于手写体中划去的文字（户口~~制度~~登记制度），DeepSeek-OCR 并没有识别到进行处理。**
## 手写体2
![示例](../../static/images/handwriter_2.jpg)
#### PaddleOCR-VL
![BD识别结果](../../static/images/handwriter_2_bd.jpg)
文档提取
```
床前明月光， 疑是地上霜。 举头望明月， 低头思故乡。
```
#### DeepSeek-OCR
文档提取
```
东前吃月光，[[75, 65, 925, 270]]
疑是地上霜。[[65, 300, 930, 490]]
举头望吃月，[[70, 520, 925, 720]]
低头思故乡。[[60, 760, 930, 936]]
```
**DeepSeek-OCR 对于较为潦草的中文手写体，识别效果并不理想。**

## 图片
![示例](../../static/images/graph.jpg)
#### PaddleOCR-VL
![BD识别结果](../../static/images/graph_bd.jpg)
文档提取
```
<div style="text-align: center;"><img src="imgs/img_in_image_box_160_55_451_488.jpg" alt="Image" width="45%" /></div>
<div style="text-align: center;">Figure 1: The Transformer - model architecture.</div>
```
#### DeepSeek-OCR
![DS识别结果](../../static/images/graph_ds.jpg)
文档提取
```
Output[[535, 103, 592, 128]]
Probabilities[[512, 130, 612, 154]]
Softmax[[535, 180, 599, 200]]
Linear[[540, 230, 590, 250]]
Add&Norm[[512, 290, 612, 310]]
Feed[[545, 323, 589, 344]]
Forward[[530, 345, 599, 369]]
Add&Norm[[512, 404, 612, 424]]
Add&Norm[[333, 428, 435, 448]]
Multi-Head[[524, 440, 610, 464]]
Feed[[370, 464, 415, 485]]
Attention[[530, 465, 600, 485]]
Forward[[360, 485, 429, 509]]
Nx[[670, 488, 694, 506]]
Add&Norm[[512, 540, 612, 560]]
Nx[[264, 555, 288, 577]]
Add&Norm[[333, 565, 435, 585]]
Masked[[535, 576, 599, 596]]
Multi-Head[[345, 599, 430, 619]]
Multi-Head[[524, 599, 610, 619]]
Attention[[355, 620, 425, 640]]
Attention[[535, 620, 604, 640]]
Positional[[249, 710, 330, 734]]
Positional[[639, 714, 715, 734]]
Encoding[[249, 736, 330, 764]]
Encoding[[639, 736, 715, 764]]
Input[[370, 771, 415, 796]]
Output[[535, 771, 599, 796]]
Embedding[[345, 793, 435, 817]]
Embedding[[524, 793, 612, 817]]
Inputs[[372, 870, 427, 895]]
Outputs[[530, 870, 600, 895]]
(shiftedright)[[511, 895, 619, 919]]
Figure1:TheTransformer-modelarchitecture.[[279, 950, 682, 974]]
```
**PaddleOCR-VL对于图片的识别处理依然很乏力，而 DeepSeek-OCR 对于图片中文字能够很好的识别提取图片中的文字。**

从上方所有场景的测评来看，除了对图片的处理，PaddleOCR-VL 在各场景下识别的准确率全面领先 DeepSeek-OCR，那么 DeepSeek-OCR 真的是徒有虚名吗？

并不是，DeepSeek-OCR 其实可以看作打着 OCR 名号的 VLM，它的优势不在于识别，而在于理解。在解释模式下，以上所有场景能够非常好完成识别-理解的全部流程并输出。甚至在解释模式下，大部分场景的识别准确率高于文字模式。

## 复杂表格

The figure presents a table comparing the performance of different models in terms of BLEU scores and training costs for English-to-German and English-to-French translations. The table is titled "Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost."

### Table Breakdown:

#### Columns:

1. **Model**: Lists the different models being compared.
2. **BLEU**: Represents the BLEU score for each model.
3. **EN-DE**: Represents the BLEU score for English-to-German translation.
4. **EN-FR**: Represents the BLEU score for English-to-French translation.
5. **Training Cost (FLOPs)**: Represents the training cost in terms of floating-point operations (FLOPs) for each model.

#### Rows:

1. **ByteNet [18]**
   - BLEU: 23.75
   - EN-DE: 39.2
   - EN-FR: 1.0 * 10^20
   - Training Cost: 1.0 * 10^20 FLOPs

2. **Deep-Att + PosUnk [39]**
   - BLEU: 24.6
   - EN-DE: 39.92
   - EN-FR: 2.3 * 10^19
   - Training Cost: 1.4 * 10^20 FLOPs

3. **GNMT + RL [38]**
   - BLEU: 25.16
   - EN-DE: 40.46
   - EN-FR: 9.6 * 10^18
   - Training Cost: 1.5 * 10^20 FLOPs

4. **ConvS2S [9]**
   - BLEU: 26.03
   - EN-DE: 40.56
   - EN-FR: 2.0 * 10^19
   - Training Cost: 1.2 * 10^20 FLOPs

5. **MoE [32]**
   - BLEU: 40.4
   - EN-DE: 41.16
   - EN-FR: 1.8 * 10^20
   - Training Cost: 1.1 * 10^21 FLOPs

6. **Deep-Att + PosUnk Ensemble [39]**
   - BLEU: 26.30
   - EN-DE: 41.16
   - EN-FR: 1.8 * 10^20
   - Training Cost: 1.1 * 10^21 FLOPs

7. **GNMT + RL Ensemble [38]**
   - BLEU: 26.36
   - EN-DE: 41.29
   - EN-FR: 7.7 * 10^19
   - Training Cost: 1.2 * 10^21 FLOPs

8. **ConvS2S Ensemble [9]**
   - BLEU: 27.3
   - EN-DE: 38.1
   - EN-FR: 3.3 * 10^18
   - Training Cost: 3.3 * 10^18 FLOPs

9. **Transformer (base model)**
   - BLEU: 28.4
   - EN-DE: 41.8
   - EN-FR: 2.3 * 10^19
   - Training Cost: 2.3 * 10^19 FLOPs

10. **Transformer (big)**
    - BLEU: 28.4
    - EN-DE: 41.8
    - EN-FR: 2.3 * 10^19
    - Training Cost: 2.3 * 10^19 FLOPs

### Analysis:

- The Transformer (big) model achieves the highest BLEU score of 28.4, indicating the best translation quality among the models compared.
- The ConvS2S Ensemble model also performs well with a BLEU score of 27.3.
- The Deep-Att + PosUnk [39] model has a BLEU score of 24.6, which is lower than the Transformer models but still competitive.
- The MoE [32] model has a BLEU score of 40.4, which is higher than the ConvS2S Ensemble model but lower than the Transformer models.
- The Deep-Att + PosUnk Ensemble [39] model has a BLEU score of 26.36, which is lower than the ConvS2S Ensemble model but higher than the MoE model.
- The GNMT + RL Ensemble [38] model has a BLEU score of 26.36, which is similar to the Deep-Att + PosUnk Ensemble model.
- The Transformer (base model) has a BLEU score of 28.4, which is slightly lower than the ConvS2S Ensemble model but still competitive.

Overall, the ConvS2S Ensemble model and the Transformer (big) model show the highest performance in terms of BLEU scores, while the Deep-Att + PosUnk [39] model also performs well. The ConvS2S Ensemble model and the Transformer (base model) have similar performance metrics.

## 图片
The figure illustrates the architecture of a Transformer model, which is a type of neural network used for processing sequential data. The Transformer model is known for its ability to handle long-range dependencies in data, making it highly effective for tasks such as machine translation, text summarization, and protein folding.

### Detailed Description:

#### Input Encoding:

- **Inputs**: The input data is fed into the model.
- **Positional Encoding**: Positional encoding is added to the input embeddings to give the model information about the order of the data points. This is because the Transformer does not inherently capture sequential information.

#### Embedding:

- **Input Embedding**: The input data is first embedded into a vector space.
- **Positional Embedding**: Positional embeddings are added to the input embeddings to retain information about the order of the data points.

#### Encoder:

- **Add & Norm**: The embeddings are added to the output of the previous layer and then normalized. This is a technique used to stabilize and speed up the training of deep neural networks.
- **Feed Forward**: The normalized output is passed through a feed-forward neural network.
- **Add & Norm**: Another normalization step is applied after the feed-forward network.
- **Multi-Head Attention**: This mechanism allows the model to focus on different parts of the input sequence for different parts of the output sequence. It involves multiple attention heads, each computing its own set of weights and outputs.
- **Add & Norm**: The output of the multi-head attention is added to the input of the attention mechanism and then normalized.
- **Feed Forward**: The result is passed through another feed-forward network.
- **Add & Norm**: Another normalization step is applied.
- **Multi-Head Attention**: This process is repeated multiple times with different weight matrices for each attention head.
- **Add & Norm**: The output of the last attention head is added to the input of the attention mechanism and then normalized.

#### Output:

- **Output Probabilities**: The final output of the Transformer model is a probability distribution over the possible outputs.
- **Softmax**: The output is passed through a softmax function to convert it into a probability distribution.
- **Linear**: The softmax output is then passed through a linear layer to produce the final output.

### Key Components:

- **Multi-Head Attention**: This mechanism allows the model to attend to different parts of the input sequence simultaneously, capturing various aspects of the data.
- **Positional Encoding**: This is crucial for the Transformer model as it helps it understand the order of the data points, which is important for tasks involving sequences.

### Summary:

The Transformer architecture, as depicted in the figure, consists of an encoder that processes input embeddings through multi-head attention mechanisms, followed by a series of normalization and feed-forward layers. The final output is a probability distribution over the possible outputs, which is used for tasks such as natural language processing, computer vision, and more. The use of positional encoding and multi-head attention enables the model to effectively capture and utilize the sequential dependencies in the data.

## 数学公式1
This image appears to display mathematical equations related to probability theory involving random variables \( X \) and \( Y \). The document is written primarily in Chinese.

### Description:

**Text Content (Chinese):**

```
题目：设有一个随机变量X服从均值为μ和方差为σ²的正态分布N(μ,σ²)。考虑二维随机变量(X,Y)，其中Y=aX+b，a≠0，b为常数。请完成以下任务：

1. 计算Y的分布及其均值和方差。
2. 设Z=X+Y，求Z的均值和方差。
3. 若X和Y的协方差为Cov(X,Y)=σXY，求Z和X的相关系数ρ(Z,X)。

解答：
1. 首先，Y的分布为线性变换后的正态分布，故Y也服从正态分布。其均值和方差为：
   \[
   E[Y] = E[aX + b] = aE[X] + b = aμ + b
   \]
   \[
   Var(Y) = Var(aX + b) = a^2Var(X) = a^2σ^2
   \]

   因此，Y ~ N(aμ + b, a^2σ^2)。
2. 对于Z=X+Y，我们可以计算其均值和方差：
   \[
   E[Z] = E[X + Y] = E[X] + E[Y] = μ + (aμ + b) = (1 + a)μ + b
   \]
   \[
   Var(Z) = Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y) = σ^2 + a^2σ^2 + 2σXY
   \]

   综上，Z的均值为(1 + a)μ + b，方差为σ^2 + a^2σ^2 + 2σXY。
3. 最后，相关系数ρ(Z,X)的计算如下：
   \[
   ρ(Z,X) = \frac{Cov(Z,X)}{\sqrt{Var(Z) \cdot Var(X)}}
   \]

   其中
   \[
   Cov(Z,X) = Cov(X + Y,X) = Cov(X,X) + Cov(Y,X) = Var(X) + Cov(Y,X) = σ^2 + σXY
   \]

   代入后，得到：
   \[
   ρ(Z,X) = \frac{σ^2 + σXY}{\sqrt{(σ^2 + a^2σ^2 + 2σXY)(σ^2)}}
   \]

   以上是本题的解答。
```

**Description Style & Aesthetic:**
The overall aesthetic suggests that this might have been taken directly off paper due to slight creases visible on what seems like standard white printer paper used commonly for academic documents.

**Foreground Elements:**

- Text primarily occupies most space within the frame without additional graphics or images aside from the mathematical expressions themselves which include various symbols representing statistical concepts:
  - \( \mu \): Mean value,
  - \( \sigma \): Standard deviation,
  - \( a \): Coefficient,
  - \( b \): Constant,

These elements collectively formulate complex mathematical relationships pertinent to probability distributions and covariance calculations typically found in advanced mathematics courses focused on statistics or probability theory.


**Background Elements:**
There's no distinct background other than plain white surface where these equations appear; hence focusing solely on the mathematical expressions presented here. 

In summary, the primary focus remains entirely on the mathematical content provided through the Chinese text describing problem-solving steps involving random variables and their associated properties. This indicates a formal educational context likely aimed towards students studying higher-level mathematics topics specifically dealing with probability theory.

## 数学公式2
The figure illustrates the concept of warmup steps in the context of optimizing a deep learning model, specifically focusing on the Adam optimizer. The image is divided into two parts:

1. **Text Explanation**:
   - The text explains that the Adam optimizer is used with specific hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.98$, and $\epsilon = 10^{-9}$.
   - It mentions that the learning rate is varied over the course of training, according to a formula provided in the paper.

2. **Mathematical Expression**:
   - The formula for the learning rate is given as:
     \[
     lrate = d_{\text{model}}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
     \]
   - This formula is used to determine the learning rate at different stages of training, incorporating both the current step number and the warmup steps.

3. **Additional Information**:
   - The caption notes that this corresponds to increasing the learning rate linearly for the first warmup steps and then decreasing it proportionally to the inverse square root of the step number.
   - The warmup steps are set to 4000.

In summary, the figure provides a mathematical explanation of how the learning rate is adjusted during the training process using the Adam optimizer, emphasizing the linear increase followed by a proportional decrease.

# 总结
从本次横向评测可以看出：

PaddleOCR-VL 依旧是“工业级 OCR”的代表，识别稳定、版面还原准确，尤其在 复杂表格与版式结构 的解析上表现出色，输出格式也更规整（HTML/Markdown 兼容度高）。

DeepSeek-OCR 则代表了下一代“理解型 OCR”，能通过不同 prompt 灵活输出结果（如 markdown、summary、QA 等），在图文内容理解与语义级任务上具有更高潜力，但在纯识别精度上仍略逊一筹。

对开发者而言，两者并非竞争，而是互补：

当需要结构化提取或文本精确还原时，选 PaddleOCR-VL；

当需要语义理解、摘要、翻译等智能任务时，选 DeepSeek-OCR。

未来 OCR 的方向将从「识字」走向「读懂」，而 DeepSeek-OCR 已经在这条路上先行一步。
如果你正在评估多模态 OCR 的落地方案，本篇的对比结论可以作为参考基准。

