import re
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Literal
from matplotlib.colors import LinearSegmentedColormap, Normalize
from .attr_dict import AttrDict

class EqualizeNormalize(Normalize):
    def __init__(self, samples, clip=False):
        super().__init__(vmin=samples.min(), vmax=samples.max(), clip=clip)
        hist, bin_edges = np.histogram(samples.flatten(), bins=256, range=(self.vmin, self.vmax), density=True)
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        self.bin_edges = bin_edges
        self.cdf = cdf
    def __call__(self, value, clip=False):
        value = np.array(value)
        return np.ma.masked_array(np.interp(value.flatten(), self.bin_edges[:-1], self.cdf).reshape(value.shape))
    def inverse(self, value):
        value = np.array(value)
        return np.ma.masked_array(np.interp(value.flatten(), self.cdf, self.bin_edges[:-1]).reshape(value.shape))

myhsv = LinearSegmentedColormap.from_list('myhsv', ['#d63031','#e17055','#fdcb6e','#00b894','#00cec9','#0984e3','#6c5ce7'], N=256)
mybwr = LinearSegmentedColormap.from_list('mybwr', ['#0984e3', '#ffffff', '#d63031'], N=256)
myhot = LinearSegmentedColormap.from_list('myhot', ['#0308F8', '#FD0B1B', '#ffff00'], gamma=5.0)

def get_fig(RN, CN, 
            FW=None, FH=None, AW=None, AH=None, A_ratio=1.0, 
            LM=5, RM=5, TM=5, BM=5, HS=None, VS=None, 
            fontsize=7, lw=0.5, gridspec=False, dpi=600, sharex=False, sharey=False):
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['figure.titlesize'] = fontsize
    plt.rcParams['lines.linewidth'] = lw
    plt.rcParams['axes.linewidth'] = lw
    plt.rcParams['xtick.major.width'] = lw
    plt.rcParams['ytick.major.width'] = lw
    plt.rcParams['xtick.minor.width'] = lw
    plt.rcParams['ytick.minor.width'] = lw
    plt.rcParams['grid.linewidth'] = lw
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    LM = LM * fontsize / 72
    RM = RM * fontsize / 72
    TM = TM * fontsize / 72
    BM = BM * fontsize / 72
    HS = HS * fontsize / 72 if HS is not None else LM
    VS = VS * fontsize / 72 if VS is not None else TM

    if AW is not None:
        AW /= 2.54
        FW = LM + CN * AW + (CN - 1) * HS + RM
        if AH is not None:
            AH /= 2.54
            FH = TM + RN * AH + (RN - 1) * VS + BM
            A_ratio = AW / AH
        else:
            AH = AW / A_ratio
            FH = TM + RN * AH + (RN - 1) * VS + BM
    elif AH is not None:
        AH /= 2.54
        AW = AH * A_ratio
        FH = TM + RN * AH + (RN - 1) * VS + BM
        FW = LM + CN * AW + (CN - 1) * HS + RM
    elif FW is not None:
        FW /= 2.54
        AW = (FW - LM - RM - (CN - 1) * HS) / CN
        if FH is not None:
            FH /= 2.54
            AH = (FH - TM - BM - (RN - 1) * VS) / RN
            A_ratio = AW / AH
        else:
            AH = AW / A_ratio
            FH = TM + RN * AH + (RN - 1) * VS + BM
    elif FH is not None:
        FH /= 2.54
        AH = (FH - TM - BM - (RN - 1) * VS) / RN
        AW = AH * A_ratio
        FW = LM + CN * AW + (CN - 1) * HS + RM
    
    figinfo = AttrDict(
        RN=RN, CN=CN, FW=FW, FH=FH, AW=AW, AH=AH, 
        LM=LM, RM=RM, TM=TM, BM=BM, HS=HS, VS=VS,
        r_AW=AW/FW, r_AH=AH/FH, r_HS=HS/FW, r_VS=VS/FH,
        r_LM=LM/FW, r_RM=RM/FW, r_TM=TM/FH, r_BM=BM/FH,
        fontsize=fontsize, lw=lw,
        top_box=(LM/FW, 1-TM/FH, 1-(LM+RM)/FW, TM/FH), 
        right_box=(1-RM/FW, BM/FH, RM/FW, 1-(TM+BM)/FH))

    if gridspec:
        fig = plt.figure(figsize=(FW, FH), dpi=dpi)
        gs = fig.add_gridspec(RN, CN, wspace=HS/AW, hspace=VS/AH, left=LM/FW, right=1-RM/FW, top=1-TM/FH, bottom=BM/FH)
        return figinfo, fig, gs
    else:
        fig, axes = plt.subplots(RN, CN, figsize=(FW, FH), dpi=dpi, sharex=sharex, sharey=sharey)
        fig.subplots_adjust(left=LM/FW, right=1-RM/FW, top=1-TM/FH, bottom=BM/FH, wspace=HS/AW, hspace=VS/AH)
        axes = axes.ravel() if RN * CN > 1 else [axes]
        return figinfo, fig, axes


def plotOD(ax, source:List[str], destination:List[str], flow:List[float], location:Dict[str, Tuple[float, float]],
           linetype:Literal['straight', 'parabola', 'rotated_parabola', 'projected_parabola']='straight', N=100, zorder=10,
           **kwargs):
    cmap = LinearSegmentedColormap.from_list('cmap', ['#0308F8', '#FD0B1B', '#ffff00'], gamma=5.0)
    norm = EqualizeNormalize(flow.values)
    t = np.linspace(0, 1, N)
    ignored = set(list(source) + list(destination)) - set(location.keys())
    if ignored: print(f'\033[33mWarning: {ignored} are not in location\033[0m')
    for (s, d, f) in zip(source, destination, flow):
        if s not in location or d not in location: continue
        p1 = np.array(location[s])[:, np.newaxis]
        p2 = np.array(location[d])[:, np.newaxis]
        if linetype == 'straight':
            ax.plot(*(p1 * (1 - t) + p2 * t), lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f))
        elif linetype == 'parabola':
            y_scale = locals().get('y_scale', np.diff(ax.get_ylim())[0])
            xy = p1 * (1-t) + p2 * t + np.array([[0], [1]]) * 4 * t * (1-t) * norm(f) * y_scale * kwargs.get('scale', 0.1)
            ax.plot(*xy, lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f))
        elif linetype == 'rotated_parabola':
            height = 0.5 * norm(f)
            C, S = (p2 - p1)[:, 0]
            A = np.array([[C, -S], [S, C]]) / 2
            if kwargs.get('adjust_up', None) and (C < 0): A = -A
            if kwargs.get('adjust_down', None) and (C > 0): A = -A
            xy = A @ np.array([2 * t - 1, height * 4 * t * (1 - t)]) + 0.5 * (p1 + p2)
            ax.plot(*xy, lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f))
        elif linetype == 'projected_parabola':
            p0 = locals().get('p0', np.array(kwargs.get('p0', [np.mean(ax.get_xlim()), np.mean(ax.get_ylim())]))[:, np.newaxis])
            D = kwargs.get('D', 10)
            xy = p0 + D * (p1 * (1-t) + p2 * t - p0) / (D - 4*t*(1 - t)*(norm(f) + 1))
            ax.plot(*xy, lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f))
        else:
            raise ValueError(f'Invalid linetype: {linetype}')
    return ax


def clear_svg(path, debug=False):
    from lxml import etree
    tree = etree.parse(path)
    root = tree.getroot()

    for text in root.findall('.//{http://www.w3.org/2000/svg}tspan') + root.findall('.//{http://www.w3.org/2000/svg}text'):
        style = text.attrib.pop('style', '')

        font_size = re.search(r'font:[^;]*\s+(\d+\.?\d*+px)(?:\s|$)', style)
        font_family = re.search(r'font:[^;]*\s+\'([^\']*)\'(?:\s|$)', style)
        font_style = re.search(r'font:[^;]*\s+(italic|oblique)(?:\s|$)', style)
        font_weight = re.search(r'font:[^;]*\s+(bold|normal|bolder|lighter)(?:\s|$)', style)
        line_height = re.search(r'line-height:\s+([\d\.]+(px|em|%)?)', style)
        font_variant = re.search(r'font-variant:\s+([\w-]+)', style)
        text_anchor = re.search(r'text-anchor:\s+([\w-]+)', style)

        if font_size: text.set('font-size', font_size.group(1))
        if font_family: text.set('font-family', font_family.group(1))
        if font_style: text.set('font-style', font_style.group(1))
        if font_weight: text.set('font-weight', font_weight.group(1))
        if line_height: text.set('line-height', line_height.group(1))
        if font_variant: text.set('font-variant', font_variant.group(1))
        if text_anchor: text.set('text-anchor', text_anchor.group(1))

        if debug: print(f'"{style}" -> "{text.attrib}"')
    tree.write(path)


def _overlapping_area(bbox1, bbox2):
    x0 = max(bbox1.x0, bbox2.x0)
    x1 = min(bbox1.x1, bbox2.x1)
    y0 = max(bbox1.y0, bbox2.y0)
    y1 = min(bbox1.y1, bbox2.y1)
    return max(0, x1 - x0) * max(0, y1 - y0)

def _distance_to_box(point, box):
    x = max(box.x0, min(point[0], box.x1))
    y = max(box.y0, min(point[1], box.y1))
    return np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)

def adjust_text(texts, ax, step=0.01, max_iterations=100, mode='xy'):
    raw_pos = np.array([text.get_position() for text in texts], dtype=float)
    cur_pos = raw_pos.copy()

    overlap = np.zeros((len(texts), len(texts)))
    def update(i, utri_only=False):
        bbox1 = texts[i].get_window_extent(renderer=ax.figure.canvas.get_renderer())
        for j, text2 in enumerate(texts):
            if j == i: continue
            if utri_only and j <= i: continue
            bbox2 = text2.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            overlap[i, j] = overlap[j, i] = _overlapping_area(bbox1, bbox2)
    
    def normalize(v):
        return v / np.linalg.norm(v).clip(1e-6)
    
    for i in range(len(texts)): update(i, utri_only=True)

    for iteration in range(max_iterations):
        i, j = random.choice(list(np.stack(np.nonzero(overlap), axis=-1)))

        F1 = (cur_pos[j] - cur_pos[i])

        bbox = texts[j].get_window_extent(renderer=ax.figure.canvas.get_renderer())
        F2 = normalize(raw_pos[j] - cur_pos[j]) * _distance_to_box(raw_pos[j], bbox) * 0.001
        
        F3 = np.random.randn(2)

        F = F1 + F2 + F3
        if mode == 'x': F[1] = 0
        if mode == 'y': F[0] = 0

        cur_pos[j] += F * step
        texts[j].set_position(cur_pos[j])
        update(j)
        if (overlap == 0).all(): break
