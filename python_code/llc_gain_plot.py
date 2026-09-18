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


def main():
    # 1. LLC 实际参数
    Lr = 20e-6       # 谐振电感 H
    Cr = 100e-9      # 谐振电容 F
    Lm = 100e-6      # 励磁电感 H
    n = 4.0          # 变压器匝比 Np/Ns
    Ro = 10.0        # 输出直流负载 Ohm

    # 2. 自动计算关键参数
    fr = 1 / (2 * np.pi * np.sqrt(Lr * Cr))
    fm = 1 / (2 * np.pi * np.sqrt((Lr + Lm) * Cr))

    Z0 = np.sqrt(Lr / Cr)
    Rac = (8 * n**2 / np.pi**2) * Ro

    Q = Z0 / Rac
    Ln = Lm / Lr

    print("========== LLC 参数 ==========")
    print(f"Lr  = {Lr * 1e6:.2f} uH")
    print(f"Lm  = {Lm * 1e6:.2f} uH")
    print(f"Cr  = {Cr * 1e9:.2f} nF")
    print(f"Ln  = {Ln:.3f}")
    print(f"fr  = {fr / 1e3:.2f} kHz")
    print(f"fm  = {fm / 1e3:.2f} kHz")
    print(f"Z0  = {Z0:.3f} ohm")
    print(f"Rac = {Rac:.3f} ohm")
    print(f"Q   = {Q:.4f}")

    # 3. 绘制不同 Q 的增益曲线
    fn = np.linspace(0.2, 2.0, 2000)
    Q_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    plt.figure(figsize=(9, 6))
    for q in Q_list:
        Mg = llc_gain(fn, q, Ln)
        plt.plot(fn, Mg, label=f"Q = {q}")

    plt.axvline(1.0, linestyle="--", linewidth=1.2, label="fn = 1")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Normalized Frequency  fn = fsw / fr")
    plt.ylabel("Gain  Mg")
    plt.title(f"LLC Gain Curves, Ln = {Ln:.2f}")
    plt.grid(True)
    plt.legend()
    plt.xlim(0.2, 2.0)
    plt.ylim(0, 3.5)
    plt.tight_layout()
    plt.show()

    # 4. 用实际开关频率绘图
    fsw = np.linspace(0.3 * fr, 2.0 * fr, 2000)
    fn_real = fsw / fr
    Mg_real = llc_gain(fn_real, Q, Ln)

    plt.figure(figsize=(9, 6))
    plt.plot(fsw / 1e3, Mg_real, linewidth=2, label="LLC Gain")
    plt.axvline(fm / 1e3, linestyle="--", label=f"fm = {fm / 1e3:.1f} kHz")
    plt.axvline(fr / 1e3, linestyle="--", label=f"fr = {fr / 1e3:.1f} kHz")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Switching Frequency fsw (kHz)")
    plt.ylabel("Gain  Mg")
    plt.title("LLC Gain vs Switching Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. 验证 fn = 1 时 Mg = 1
    Mg_test = llc_gain(1.0, Q, Ln)
    print()
    print("========== 验证 ==========")
    print(f"fn = 1 时 Mg = {Mg_test:.6f}")
    print("理论上应接近 1")


if __name__ == "__main__":
    main()
