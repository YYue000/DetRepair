import imagecorruptions


def output_table(results, table_label, _ap_str, clean_tab=False):
    MAXSERVIRITY, MINSERVIRITY = 3, 3
    SEVIRITY_NUM = MAXSERVIRITY-MINSERVIRITY+1

    outs = '\multicolumn{2}{|c|}{model}'
    if clean_tab:
        outs += '&\\rotatebox{270}{clean}'
    for i,c in enumerate(imagecorruptions.get_corruption_names()):
        outs += '&\\rotatebox{270}{'+c.split('_')[0]+'}'

    MAXC = i+1
    outs += '\\\\\n\\hline\n'

    for m, minfo in results.items():
        outs += '\\multirow{'+str(SEVIRITY_NUM)+'}*{'+m.replace('_','\_')+'}'
        if clean_tab:
            _clean_str = _ap_str(minfo.get('clean', None))
        for s in range(MINSERVIRITY, MAXSERVIRITY+1):
            outs += f'&{s}'
            if clean_tab:
                if s == 1:
                    outs += _clean_str
                else:
                    outs += _ap_str(None)
            for c in imagecorruptions.get_corruption_names():
                ap = minfo.get(c, {}).get(s, None)
                outs += _ap_str(ap)
            outs += '\\\\\n'
        outs += '\\hline\n'

    if clean_tab:
        col = MAXC+3
    else:
        col = MAXC+2
    outs = '\\begin{table}[htbp]\n\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{|'+'c|'*col+'}\n\\hline\n' + outs + '\end{tabular}\n}\n\caption{'+table_label+'}\n\label{tab:'+table_label+'}\n\end{table}'
    return outs


