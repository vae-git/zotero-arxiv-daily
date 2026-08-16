"""RF/PA terminology used to constrain LLM translations.

A general-purpose model left to its own devices renders "Doherty" as a literal
transliteration, "back-off" as "后退", and "outphasing" as anything at all. Feeding
it the accepted Chinese term for the vocabulary that actually appears in a given
paper keeps titles and abstracts readable to an RF engineer.

Only the entries that occur in the text are injected into the prompt (see
:func:`glossary_hint_for`), so the token cost stays proportional to how much RF
jargon a paper actually contains.
"""

# 英文术语 → 约定中文译法。键按原始大小写书写用于展示，匹配时统一转小写。
RF_GLOSSARY: dict[str, str] = {
    # 器件与拓扑
    "power amplifier": "功率放大器",
    "Doherty": "多赫蒂",
    "outphasing": "异相合成",
    "envelope tracking": "包络跟踪",
    "load modulation": "负载调制",
    "stacked": "堆叠",
    "balun": "巴伦",
    "Wilkinson": "威尔金森",
    "transformer": "变压器",
    "class-F": "F 类",
    "class-E": "E 类",
    "class-J": "J 类",
    "continuous mode": "连续模式",
    "switch-mode": "开关模式",
    "low noise amplifier": "低噪声放大器",
    "LNA": "低噪声放大器",
    "phased array": "相控阵",
    "beamforming": "波束成形",
    "antenna": "天线",
    "transceiver": "收发机",
    "transmitter": "发射机",
    "rectifier": "整流器",
    # 线性化与建模
    "digital predistortion": "数字预失真",
    "predistortion": "预失真",
    "linearization": "线性化",
    "linearity": "线性度",
    "memory effect": "记忆效应",
    "memory polynomial": "记忆多项式",
    "Volterra series": "沃尔泰拉级数",
    "behavioral model": "行为模型",
    "intermodulation": "互调",
    "crest factor reduction": "峰均比抑制",
    "peak-to-average power ratio": "峰均功率比",
    "PAPR": "峰均功率比",
    # 指标
    "power added efficiency": "功率附加效率",
    "PAE": "功率附加效率",
    "drain efficiency": "漏极效率",
    "back-off": "功率回退",
    "gain compression": "增益压缩",
    "adjacent channel leakage ratio": "邻道泄漏功率比",
    "ACLR": "邻道泄漏功率比",
    "error vector magnitude": "误差矢量幅度",
    "EVM": "误差矢量幅度",
    "third-order intercept": "三阶交调截点",
    "bandwidth": "带宽",
    "saturation": "饱和",
    # 设计方法
    "load-pull": "负载牵引",
    "source-pull": "源牵引",
    "harmonic tuning": "谐波调谐",
    "harmonic balance": "谐波平衡",
    "impedance matching": "阻抗匹配",
    "matching network": "匹配网络",
    "S-parameter": "S 参数",
    "Smith chart": "史密斯圆图",
    "quadrature": "正交",
    "duty cycle": "占空比",
    # 工艺与频段
    "gallium nitride": "氮化镓",
    "GaN": "氮化镓",
    "gallium arsenide": "砷化镓",
    "GaAs": "砷化镓",
    "HEMT": "高电子迁移率晶体管",
    "LDMOS": "横向扩散金属氧化物半导体",
    "MMIC": "单片微波集成电路",
    "millimeter-wave": "毫米波",
    "microwave": "微波",
    "terahertz": "太赫兹",
    "waveguide": "波导",
    "microstrip": "微带",
    "resonator": "谐振器",
}

# 这些记号必须原样保留，翻译成中文反而看不懂。
RF_KEEP_AS_IS: tuple[str, ...] = (
    "AM-AM",
    "AM-PM",
    "GaN",
    "GaAs",
    "CMOS",
    "SiGe",
    "MMIC",
    "HEMT",
    "LDMOS",
    "PAE",
    "ACLR",
    "EVM",
    "PAPR",
    "dBm",
    "dBc",
    "GHz",
    "MHz",
)


def glossary_hint_for(*texts: str | None) -> str:
    """Return a prompt fragment listing only the glossary terms present in *texts*.

    Returns an empty string when the text contains no known RF terminology, so
    non-RF papers do not pay the token cost.
    """
    blob = " ".join(text or "" for text in texts).lower()
    if not blob.strip():
        return ""

    hits = [(term, zh) for term, zh in RF_GLOSSARY.items() if term.lower() in blob]
    if not hits:
        return ""

    lines = "\n".join(f"- {term} → {zh}" for term, zh in hits)
    keep = ", ".join(token for token in RF_KEEP_AS_IS if token.lower() in blob)
    hint = (
        "Use these established Chinese renderings for the RF terminology in this paper:\n"
        f"{lines}\n"
    )
    if keep:
        hint += f"Keep these tokens exactly as-is, do not translate them: {keep}\n"
    return hint
