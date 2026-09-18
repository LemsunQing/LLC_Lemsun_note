"""
LLC FHA 增益曲线 + 感性/容性工作区分界分析

功能说明
--------
1. 保留原来的 LLC FHA 增益计算：
       Mg = Mg(fn, Q, Ln)

2. 新增 LLC 输入阻抗虚部计算：
       Xn = fn - 1/fn + fn*Ln / (1 + (fn*Ln*Q)^2)

   判定规则：
       Xn < 0  -> 容性区 Capacitive Region
       Xn = 0  -> 感容分界
       Xn > 0  -> 感性区 Inductive Region

3. 对每一个 Q（不同负载）求出 Xn = 0 对应的分界频率 fn_boundary。

4. 在“不同 Q 的增益曲线族”上：
   - 标出每条曲线的感容分界点；
   - 将所有分界点连续连接，形成“感性/容性分界曲线”。

5. 额外绘制一个 Q-fn 工作区图：
   - 红色区域：容性区；
   - 绿色区域：感性区；
   - 黑色曲线：随负载 Q 变化的感容分界线。
   这张图最适合直接观察“负载变化导致感容边界移动”。

6. 同时标出 LLC 两个特征频率：
       fm = 1 / (2*pi*sqrt((Lr+Lm)*Cr))
       fr = 1 / (2*pi*sqrt(Lr*Cr))

注意
----
- Q 越大，在本文定义下表示负载越重。
- fm 本身不是所有负载下固定的感容边界；
  真正边界还会随 Q 变化。
- 本程序使用 FHA（Fundamental Harmonic Approximation）模型。
"""

import numpy as np
import matplotlib.pyplot as plt


def llc_gain(fn, Q, Ln):
    """
    LLC FHA 增益模型

    fn = fsw / fr
    Q  = sqrt(Lr / Cr) / Rac
    Ln = Lm / Lr
    """
    fn = np.asarray(fn)

    real_part = 1 + 1 / Ln - 1 / (Ln * fn**2)
    imag_part = Q * (fn - 1 / fn)

    return 1 / np.sqrt(real_part**2 + imag_part**2)


def llc_input_reactance_norm(fn, Q, Ln):
    """
    LLC 输入阻抗归一化虚部 Xn

    Xn < 0 : 容性区
    Xn = 0 : 感容分界
    Xn > 0 : 感性区
    """
    fn = np.asarray(fn)

    return (
        fn
        - 1 / fn
        + (fn * Ln) / (1 + (fn * Ln * Q)**2)
    )


def llc_boundary_fn(Q, Ln):
    """
    求感性区 / 容性区的归一化分界频率 fn_boundary。

    从 Xn = 0 出发，令 y = fn^2，可整理为：

        (Ln^2 * Q^2) * y^2
        + (1 + Ln - Ln^2 * Q^2) * y
        - 1 = 0

    取正根后：
        fn_boundary = sqrt(y)
    """
    Q = np.asarray(Q, dtype=float)

    # 支持标量和数组
    result = np.empty_like(Q, dtype=float)

    nearly_zero = np.abs(Q) < 1e-12
    result[nearly_zero] = 1 / np.sqrt(1 + Ln)

    q = Q[~nearly_zero]

    if q.size > 0:
        a = Ln**2 * q**2
        b = 1 + Ln - a

        y = (
            -b
            + np.sqrt(b**2 + 4 * a)
        ) / (2 * a)

        result[~nearly_zero] = np.sqrt(y)

    if result.ndim == 0:
        return float(result)

    return result


def main():
    # ============================================================
    # 1. LLC 实际参数
    #    保留原脚本参数，后续只需修改这里即可
    # ============================================================

    Lr = 20e-6       # 谐振电感 [H]
    Cr = 100e-9      # 谐振电容 [F]
    Lm = 100e-6      # 励磁电感 [H]

    n = 4.0          # 变压器匝比 Np/Ns
    Ro = 10.0        # 输出直流负载 [ohm]


    # ============================================================
    # 2. 自动计算关键参数
    # ============================================================

    fr = 1 / (2 * np.pi * np.sqrt(Lr * Cr))

    fm = 1 / (
        2 * np.pi
        * np.sqrt((Lr + Lm) * Cr)
    )

    Z0 = np.sqrt(Lr / Cr)

    Rac = (
        8 * n**2 / np.pi**2
    ) * Ro

    Q_actual = Z0 / Rac

    Ln = Lm / Lr

    fn_m = fm / fr


    print("========== LLC 参数 ==========")
    print(f"Lr       = {Lr * 1e6:.2f} uH")
    print(f"Lm       = {Lm * 1e6:.2f} uH")
    print(f"Cr       = {Cr * 1e9:.2f} nF")
    print(f"Ln       = {Ln:.3f}")
    print(f"fr       = {fr / 1e3:.2f} kHz")
    print(f"fm       = {fm / 1e3:.2f} kHz")
    print(f"fm/fr    = {fn_m:.4f}")
    print(f"Z0       = {Z0:.3f} ohm")
    print(f"Rac      = {Rac:.3f} ohm")
    print(f"Q(actual)= {Q_actual:.4f}")


    # ============================================================
    # 3. 不同负载 Q 的增益曲线
    # ============================================================

    fn = np.linspace(
        0.2,
        2.0,
        3000
    )

    # Q 越大 -> 负载越重
    Q_list = np.array([
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ])


    plt.figure(
        figsize=(10.5, 7)
    )

    # ------------------------------------------------------------
    # 绘制不同 Q 下的增益曲线，并标出各自的感容边界点
    # ------------------------------------------------------------

    for q in Q_list:

        Mg = llc_gain(
            fn,
            q,
            Ln
        )

        fn_b = llc_boundary_fn(
            q,
            Ln
        )

        Mg_b = llc_gain(
            fn_b,
            q,
            Ln
        )

        plt.plot(
            fn,
            Mg,
            linewidth=1.7,
            label=f"Q = {q:.1f}"
        )

        plt.scatter(
            fn_b,
            Mg_b,
            s=38,
            zorder=5
        )


    # ------------------------------------------------------------
    # 4. 生成连续的“感性 / 容性分界曲线”
    #
    #    对连续 Q 扫描：
    #       Q -> fn_boundary -> Mg_boundary
    #
    #    然后将这些边界点连接起来。
    # ------------------------------------------------------------

    Q_boundary = np.linspace(
        Q_list.min(),
        Q_list.max(),
        500
    )

    fn_boundary_curve = llc_boundary_fn(
        Q_boundary,
        Ln
    )

    Mg_boundary_curve = np.array([
        llc_gain(fn_b, q, Ln)
        for fn_b, q in zip(
            fn_boundary_curve,
            Q_boundary
        )
    ])


    plt.plot(
        fn_boundary_curve,
        Mg_boundary_curve,
        linewidth=3.0,
        label="Capacitive / Inductive Boundary"
    )


    # ------------------------------------------------------------
    # 标出两个 LLC 特征频率
    # ------------------------------------------------------------

    plt.axvline(
        fn_m,
        linestyle=":",
        linewidth=1.8,
        label=f"fm/fr = {fn_m:.3f}"
    )

    plt.axvline(
        1.0,
        linestyle="--",
        linewidth=1.8,
        label="fr/fr = 1.000"
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0
    )


    plt.text(
        0.26,
        0.38,
        "Capacitive side",
        fontsize=11
    )

    plt.text(
        1.30,
        0.38,
        "Inductive side",
        fontsize=11
    )


    plt.xlabel(
        "Normalized Frequency  fn = fsw / fr"
    )

    plt.ylabel(
        "Gain  Mg"
    )

    plt.title(
        f"LLC Gain Family and Capacitive / Inductive Boundary, Ln = {Ln:.2f}"
    )

    plt.grid(True)

    plt.legend()

    plt.xlim(
        0.2,
        2.0
    )

    plt.ylim(
        0,
        5.5
    )

    plt.tight_layout()

    plt.show()


    # ============================================================
    # 5. Q-fn 工作区图
    #
    #    这一张图是“感性 / 容性区域划分”最清晰的表达。
    #
    #    横轴：归一化频率 fn
    #    纵轴：品质因数 Q（负载程度）
    #
    #    分界曲线左侧 = 容性
    #    分界曲线右侧 = 感性
    # ============================================================

    Q_map = np.linspace(
        0.05,
        1.05,
        600
    )

    fn_boundary_map = llc_boundary_fn(
        Q_map,
        Ln
    )


    plt.figure(
        figsize=(10.5, 6.5)
    )


    # ------------------------------------------------------------
    # 容性区背景
    # ------------------------------------------------------------

    plt.fill_betweenx(
        Q_map,
        0.2,
        fn_boundary_map,
        alpha=0.30,
        label="Capacitive Region"
    )


    # ------------------------------------------------------------
    # 感性区背景
    # ------------------------------------------------------------

    plt.fill_betweenx(
        Q_map,
        fn_boundary_map,
        2.0,
        alpha=0.25,
        label="Inductive Region"
    )


    # ------------------------------------------------------------
    # 真正的感容分界曲线 Xn = 0
    # ------------------------------------------------------------

    plt.plot(
        fn_boundary_map,
        Q_map,
        linewidth=3.0,
        label="Boundary: Im(Zin) = 0"
    )


    # ------------------------------------------------------------
    # 标出前面 Q_list 对应的边界工作点
    # ------------------------------------------------------------

    fn_boundary_points = llc_boundary_fn(
        Q_list,
        Ln
    )

    plt.scatter(
        fn_boundary_points,
        Q_list,
        s=42,
        zorder=5,
        label="Selected load points"
    )


    # ------------------------------------------------------------
    # 标出两个谐振点
    # ------------------------------------------------------------

    plt.axvline(
        fn_m,
        linestyle=":",
        linewidth=1.8,
        label=f"fm/fr = {fn_m:.3f}"
    )

    plt.axvline(
        1.0,
        linestyle="--",
        linewidth=1.8,
        label="fr/fr = 1.000"
    )


    plt.text(
        0.28,
        0.72,
        "Capacitive",
        fontsize=12
    )

    plt.text(
        1.25,
        0.72,
        "Inductive",
        fontsize=12
    )


    plt.xlabel(
        "Normalized Frequency  fn = fsw / fr"
    )

    plt.ylabel(
        "Quality Factor Q  (larger Q = heavier load)"
    )

    plt.title(
        f"LLC Capacitive / Inductive Operating Map, Ln = {Ln:.2f}"
    )

    plt.grid(True)

    plt.legend()

    plt.xlim(
        0.2,
        2.0
    )

    plt.ylim(
        Q_map.min(),
        Q_map.max()
    )

    plt.tight_layout()

    plt.show()


    # ============================================================
    # 6. 打印各 Q 对应的感容边界
    # ============================================================

    print()
    print("========== 感容边界 ==========")

    for q in Q_list:

        fn_b = llc_boundary_fn(
            q,
            Ln
        )

        print(
            f"Q = {q:.2f}  ->  "
            f"fn_boundary = {fn_b:.4f}  ->  "
            f"f_boundary = {fn_b * fr / 1e3:.2f} kHz"
        )


    # ============================================================
    # 7. 原脚本实际参数对应的工作区判断
    # ============================================================

    fn_actual_boundary = llc_boundary_fn(
        Q_actual,
        Ln
    )

    print()
    print("========== 当前实际参数 ==========")

    print(
        f"Q_actual = {Q_actual:.4f}"
    )

    print(
        f"Boundary fn = {fn_actual_boundary:.4f}"
    )

    print(
        f"Boundary frequency = "
        f"{fn_actual_boundary * fr / 1e3:.2f} kHz"
    )

    print(
        f"Capacitive: fn < {fn_actual_boundary:.4f}"
    )

    print(
        f"Inductive : fn > {fn_actual_boundary:.4f}"
    )


if __name__ == "__main__":
    main()
