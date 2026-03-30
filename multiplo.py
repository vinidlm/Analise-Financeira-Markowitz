"""
APLICAÇÃO STREAMLIT - ANÁLISE FINANCEIRA COM MÚLTIPLAS AÇÕES + MARKOWITZ
Versão expandida: suporta análise de 1 ou mais ações, gráficos comparativos e Markowitz
Mantém a estrutura do original (IA, métricas, gráficos) e adiciona análise de Markowitz
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAÇÕES INICIAIS
# ============================================================

st.set_page_config(page_title="Análise Financeira + Markowitz", layout="wide")

TAXA_LIVRE_RISCO = 0.05
DIAS_UTEIS_ANO = 252


# ============================================================
# MODELO DE IA
# ============================================================

@st.cache_resource
def carregar_modelo_classificacao():
    """Carrega modelo zero-shot de forma lazy."""
    try:
        from transformers import pipeline

        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
    except Exception as e:
        st.warning(f"⚠️ Modelo de IA indisponível: {e}")
        return None


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def garantir_serie_1d(obj, nome: str | None = None) -> pd.Series:
    """Garante que o objeto vire uma Series 1D consistente."""
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0].copy()
        else:
            s = obj.iloc[:, 0].copy()
    else:
        s = pd.Series(obj)

    if nome is not None:
        s = s.rename(nome)
    return s


def parse_data_br(data_str: str) -> datetime:
    """Converte DD-MM-YYYY para datetime."""
    return datetime.strptime(data_str, "%d-%m-%Y")


def anualizar_retorno(retorno_diario_medio: float) -> float:
    """Anualiza retorno diário."""
    return (1 + retorno_diario_medio) ** DIAS_UTEIS_ANO - 1


def anualizar_volatilidade(vol_diaria: float) -> float:
    """Anualiza volatilidade diária."""
    return vol_diaria * np.sqrt(DIAS_UTEIS_ANO)


def calcular_beta(retornos_ativo: pd.Series, retornos_mercado: pd.Series) -> float:
    """Calcula beta."""
    try:
        cov = np.cov(retornos_ativo, retornos_mercado)[0, 1]
        var_mercado = np.var(retornos_mercado, ddof=1)
        return cov / var_mercado if var_mercado != 0 else 0
    except Exception:
        return np.nan


def calcular_capm(taxa_livre_risco: float, retorno_mercado: float, beta: float) -> float:
    """Calcula retorno esperado pelo CAPM."""
    return taxa_livre_risco + beta * (retorno_mercado - taxa_livre_risco)


def calcular_sharpe(retorno_anual: float, vol_anual: float) -> float:
    """Calcula Sharpe Ratio."""
    if vol_anual == 0 or pd.isna(vol_anual):
        return 0
    return (retorno_anual - TAXA_LIVRE_RISCO) / vol_anual


def calcular_sortino(retorno_diario: pd.Series) -> float:
    """Calcula Sortino Ratio."""
    retorno_anual = anualizar_retorno(retorno_diario.mean())
    downside = retorno_diario[retorno_diario < 0].std(ddof=1)

    if pd.isna(downside) or downside == 0:
        return 0

    vol_downside_anual = downside * np.sqrt(DIAS_UTEIS_ANO)
    return (retorno_anual - TAXA_LIVRE_RISCO) / vol_downside_anual


def calcular_max_drawdown(preco: pd.Series) -> tuple[float, pd.Series]:
    """Calcula max drawdown e série de drawdown."""
    cummax = preco.expanding().max()
    drawdown = (preco - cummax) / cummax
    max_dd = drawdown.min()
    return max_dd, drawdown


@st.cache_data(show_spinner=False)
def baixar_dados(ticker: str, inicio_br: str, fim_br: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa dados do Yahoo Finance e prepara retorno diário."""
    try:
        inicio = parse_data_br(inicio_br)
        fim_inclusivo = parse_data_br(fim_br)
        fim_exclusivo = fim_inclusivo + timedelta(days=1)

        df = yf.download(
            ticker,
            start=inicio.strftime("%Y-%m-%d"),
            end=fim_exclusivo.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return None, None

        coluna_preco = "Adj Close" if "Adj Close" in df.columns else "Close"
        proc_df = df[[coluna_preco]].copy()
        proc_df.rename(columns={coluna_preco: "preco"}, inplace=True)
        proc_df["retorno_diario"] = proc_df["preco"].pct_change()
        proc_df.dropna(inplace=True)

        return df, proc_df

    except Exception as e:
        st.error(f"❌ Erro ao baixar dados de {ticker}: {e}")
        return None, None


# ============================================================
# FUNÇÕES FINANCEIRAS PRINCIPAIS
# ============================================================

def calcular_metricas_completas(
    ticker_ativo: str,
    ticker_mercado: str,
    inicio_br: str,
    fim_br: str
) -> dict | None:
    """Calcula todas as métricas financeiras do ativo e do mercado."""
    _, ativo = baixar_dados(ticker_ativo, inicio_br, fim_br)
    _, mercado = baixar_dados(ticker_mercado, inicio_br, fim_br)

    if ativo is None or mercado is None:
        return None

    dados_alinhados = pd.concat(
        [
            ativo["retorno_diario"].rename("ativo"),
            mercado["retorno_diario"].rename("mercado")
        ],
        axis=1,
        join="inner"
    ).dropna()

    if dados_alinhados.empty:
        return None

    ret_ativo = dados_alinhados["ativo"]
    ret_mercado = dados_alinhados["mercado"]

    retorno_anual_ativo = anualizar_retorno(ret_ativo.mean())
    vol_anual_ativo = anualizar_volatilidade(ret_ativo.std(ddof=1))

    retorno_anual_mercado = anualizar_retorno(ret_mercado.mean())
    vol_anual_mercado = anualizar_volatilidade(ret_mercado.std(ddof=1))

    beta = calcular_beta(ret_ativo, ret_mercado)
    capm = calcular_capm(TAXA_LIVRE_RISCO, retorno_anual_mercado, beta)
    sharpe = calcular_sharpe(retorno_anual_ativo, vol_anual_ativo)
    sortino = calcular_sortino(ret_ativo)

    max_dd, dd_series = calcular_max_drawdown(ativo["preco"])

    alfa = retorno_anual_ativo - capm
    retorno_excedente = retorno_anual_ativo - retorno_anual_mercado
    bateu_mercado = retorno_excedente > 0

    pior_data_dd = dd_series.idxmin()
    pior_dd = dd_series.min()

    return {
        "ticker": ticker_ativo,
        "ticker_mercado": ticker_mercado,
        "periodo_dias": int(len(ret_ativo)),
        "retorno_anual": float(retorno_anual_ativo),
        "volatilidade_anual": float(vol_anual_ativo),
        "retorno_mercado": float(retorno_anual_mercado),
        "volatilidade_mercado": float(vol_anual_mercado),
        "beta": float(beta),
        "capm": float(capm),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "alfa": float(alfa),
        "retorno_excedente": float(retorno_excedente),
        "bateu_mercado": bool(bateu_mercado),
        "pior_data_dd": pior_data_dd,
        "pior_dd": float(pior_dd),
        "retorno_diario": ret_ativo,
        "retorno_diario_mercado": ret_mercado,
        "precos": ativo["preco"],
        "precos_mercado": mercado["preco"],
        "dd_series": dd_series,
    }


# ============================================================
# FUNÇÕES DE MARKOWITZ
# ============================================================

def construir_dataframe_retornos(todas_metricas: dict) -> pd.DataFrame:
    """Constrói DataFrame com retornos diários alinhados de todas as ações."""
    retornos_dict = {}
    
    for ticker, metricas in todas_metricas.items():
        retornos_dict[ticker] = metricas["retorno_diario"]
    
    df_retornos = pd.concat(retornos_dict, axis=1, join="inner").dropna()
    return df_retornos


def calcular_matriz_correlacao(df_retornos: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de correlação entre as ações."""
    return df_retornos.corr()


def calcular_matriz_covariancia(df_retornos: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de covariância anualizada."""
    cov_diaria = df_retornos.cov()
    return cov_diaria * DIAS_UTEIS_ANO


def retorno_portfólio(pesos: np.ndarray, retornos_anuais: np.ndarray) -> float:
    """Calcula retorno esperado do portfólio."""
    return np.sum(pesos * retornos_anuais)


def volatilidade_portfólio(pesos: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calcula volatilidade (desvio padrão) do portfólio."""
    return np.sqrt(np.dot(pesos, np.dot(cov_matrix, pesos)))


def sharpe_portfólio(pesos: np.ndarray, retornos_anuais: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calcula Sharpe Ratio do portfólio."""
    ret = retorno_portfólio(pesos, retornos_anuais)
    vol = volatilidade_portfólio(pesos, cov_matrix)
    return (ret - TAXA_LIVRE_RISCO) / vol if vol > 0 else 0


def otimizar_sharpe_maximo(retornos_anuais: np.ndarray, cov_matrix: np.ndarray, tickers: list) -> dict:
    """Encontra o portfólio com máximo Sharpe Ratio."""
    n_ativos = len(tickers)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_ativos))
    x0 = np.array([1 / n_ativos] * n_ativos)

    result = minimize(
        lambda x: -sharpe_portfólio(x, retornos_anuais, cov_matrix),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        pesos_otimos = result.x
        ret_otimo = retorno_portfólio(pesos_otimos, retornos_anuais)
        vol_otimo = volatilidade_portfólio(pesos_otimos, cov_matrix)
        sharpe_otimo = sharpe_portfólio(pesos_otimos, retornos_anuais, cov_matrix)

        return {
            "pesos": pesos_otimos,
            "retorno": ret_otimo,
            "volatilidade": vol_otimo,
            "sharpe": sharpe_otimo,
            "sucesso": True
        }
    else:
        return {
            "pesos": np.array([1 / n_ativos] * n_ativos),
            "retorno": None,
            "volatilidade": None,
            "sharpe": None,
            "sucesso": False
        }


def gerar_fronteira_eficiente(
    retornos_anuais: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list,
    num_portfolios: int = 100
) -> tuple[list, list, list]:
    """Gera múltiplos portfólios aleatórios."""
    n_ativos = len(tickers)
    retornos_lista = []
    volatilidades_lista = []
    sharpes_lista = []

    np.random.seed(42)
    for _ in range(num_portfolios):
        pesos = np.random.dirichlet(np.ones(n_ativos))
        ret = retorno_portfólio(pesos, retornos_anuais)
        vol = volatilidade_portfólio(pesos, cov_matrix)
        sharpe = sharpe_portfólio(pesos, retornos_anuais, cov_matrix)

        retornos_lista.append(ret)
        volatilidades_lista.append(vol)
        sharpes_lista.append(sharpe)

    return retornos_lista, volatilidades_lista, sharpes_lista


# ============================================================
# ANÁLISE / IA / INTERPRETAÇÃO
# ============================================================

def classificar_risco(metricas: dict) -> str:
    """Classifica o risco total do ativo."""
    vol = float(metricas["volatilidade_anual"])
    max_dd = float(metricas["max_drawdown"])
    beta = float(metricas["beta"])

    score_risco = 0

    if vol > 0.40:
        score_risco += 2
    elif vol > 0.25:
        score_risco += 1

    if max_dd < -0.40:
        score_risco += 2
    elif max_dd < -0.20:
        score_risco += 1

    if beta > 1.3:
        score_risco += 1

    if score_risco >= 4:
        return "ALTO"
    elif score_risco >= 2:
        return "MODERADO"
    return "BAIXO"


def classificar_horizonte(metricas: dict) -> str:
    """Classifica horizonte mais adequado com base nos dados."""
    sharpe = float(metricas["sharpe"])
    retorno = float(metricas["retorno_anual"])
    max_dd = float(metricas["max_drawdown"])
    vol = float(metricas["volatilidade_anual"])

    if max_dd < -0.50:
        return "CURTO PRAZO"

    if sharpe > 0.7 and retorno > 0 and max_dd > -0.25 and vol <= 0.45:
        return "LONGO PRAZO"

    return "MÉDIO PRAZO"


def classificar_perfil_investidor(risco: str) -> str:
    """Relaciona o nível de risco com o perfil do investidor."""
    if risco == "ALTO":
        return "AGRESSIVO"
    elif risco == "MODERADO":
        return "MODERADO"
    return "CONSERVADOR"


def comparar_com_mercado(metricas: dict) -> str:
    """Compara desempenho do ativo com o mercado."""
    retorno_excedente = float(metricas["retorno_excedente"])
    if retorno_excedente > 0:
        return "SUPEROU O MERCADO"
    elif retorno_excedente < 0:
        return "FICOU ABAIXO DO MERCADO"
    return "IGUAL AO MERCADO"


def calcular_score_financeiro(metricas: dict) -> int:
    """Gera score quantitativo baseado em métricas financeiras."""
    sharpe = float(metricas["sharpe"])
    retorno = float(metricas["retorno_anual"])
    max_dd = float(metricas["max_drawdown"])
    alfa = float(metricas["alfa"])
    beta = float(metricas["beta"])

    score = 0

    if sharpe > 1:
        score += 3
    elif sharpe > 0.5:
        score += 2
    elif sharpe > 0:
        score += 1
    else:
        score -= 2

    if retorno > 0.15:
        score += 2
    elif retorno > 0:
        score += 1
    else:
        score -= 2

    if max_dd > -0.2:
        score += 2
    elif max_dd > -0.4:
        score += 1
    else:
        score -= 2

    if alfa > 0:
        score += 2
    else:
        score -= 1

    if beta > 1.5:
        score -= 1

    return score


def gerar_texto_analise(metricas: dict) -> str:
    """Gera resumo textual para o modelo de IA."""
    ticker = metricas["ticker"]
    ret = float(metricas["retorno_anual"])
    vol = float(metricas["volatilidade_anual"])
    sharpe = float(metricas["sharpe"])
    beta = float(metricas["beta"])
    max_dd = float(metricas["max_drawdown"])
    alfa = float(metricas["alfa"])
    risco = classificar_risco(metricas)
    horizonte = classificar_horizonte(metricas)
    comparacao = comparar_com_mercado(metricas)

    texto = f"""
Ativo: {ticker}
Retorno anual: {ret*100:.2f}%
Volatilidade anual: {vol*100:.2f}%
Sharpe Ratio: {sharpe:.3f}
Beta: {beta:.3f}
Maximum Drawdown: {max_dd*100:.2f}%
Alfa: {alfa*100:.2f}%
Risco total: {risco}
Horizonte indicado: {horizonte}
Comparação com o mercado: {comparacao}
"""
    return texto.strip()


def analisar_recomendacao_completa(modelo, metricas: dict) -> dict:
    """Recomendação baseada em score quantitativo."""
    try:
        score = calcular_score_financeiro(metricas)

        if score >= 5:
            recomendacao = "COMPRAR"
        elif score <= 1:
            recomendacao = "VENDER"
        else:
            recomendacao = "MANTER"

        confianca = min(abs(score) / 8, 1.0)

        if score >= 5:
            sentimento = "POSITIVO"
        elif score <= 1:
            sentimento = "NEGATIVO"
        else:
            sentimento = "NEUTRO"

        risco = classificar_risco(metricas)
        horizonte = classificar_horizonte(metricas)
        perfil = classificar_perfil_investidor(risco)
        comparacao = comparar_com_mercado(metricas)

        ia_interpretacao = None
        ia_top_label = None
        ia_top_score = None

        if modelo is not None:
            texto = gerar_texto_analise(metricas)
            resultado = modelo(
                texto,
                candidate_labels=["good investment", "neutral", "bad investment"],
                multi_label=False
            )
            ia_top_label = resultado["labels"][0]
            ia_top_score = float(resultado["scores"][0])
            ia_interpretacao = resultado

        return {
            "recomendacao": recomendacao,
            "confianca": float(confianca),
            "sentimento": sentimento,
            "score": int(score),
            "risco": risco,
            "horizonte": horizonte,
            "perfil": perfil,
            "comparacao_mercado": comparacao,
            "ia_top_label": ia_top_label,
            "ia_top_score": ia_top_score,
            "ia_interpretacao": ia_interpretacao,
        }

    except Exception as e:
        st.warning(f"⚠️ Erro na análise: {e}")

        risco = classificar_risco(metricas)
        horizonte = classificar_horizonte(metricas)
        perfil = classificar_perfil_investidor(risco)
        comparacao = comparar_com_mercado(metricas)

        sharpe = float(metricas["sharpe"])
        max_dd = float(metricas["max_drawdown"])
        retorno = float(metricas["retorno_anual"])

        if retorno > 0 and sharpe > 0.5 and max_dd > -0.3:
            rec = "COMPRAR"
            sent = "POSITIVO"
            score = 5
        elif retorno < 0 or max_dd < -0.5:
            rec = "VENDER"
            sent = "NEGATIVO"
            score = 1
        else:
            rec = "MANTER"
            sent = "NEUTRO"
            score = 3

        return {
            "recomendacao": rec,
            "confianca": 0.6,
            "sentimento": sent,
            "score": score,
            "risco": risco,
            "horizonte": horizonte,
            "perfil": perfil,
            "comparacao_mercado": comparacao,
            "ia_top_label": None,
            "ia_top_score": None,
            "ia_interpretacao": "Análise por regras (IA indisponível)",
        }


def gerar_interpretacao_ia(metricas: dict, resultado: dict) -> str:
    """Texto final com foco em risco, retorno, horizonte e comparação com mercado."""
    ticker = metricas["ticker"]
    ret = float(metricas["retorno_anual"])
    vol = float(metricas["volatilidade_anual"])
    sharpe = float(metricas["sharpe"])
    max_dd = float(metricas["max_drawdown"])
    alfa = float(metricas["alfa"])
    retorno_excedente = float(metricas["retorno_excedente"])

    rec = resultado["recomendacao"]
    conf = float(resultado["confianca"])
    score = resultado["score"]
    risco = resultado["risco"]
    horizonte = resultado["horizonte"]
    perfil = resultado["perfil"]
    comparacao = resultado["comparacao_mercado"]

    interpretacao = f"""
🤖 **ANÁLISE DE RISCO E RETORNO**

**Ativo:** {ticker}

**Resumo numérico**
- Retorno anual: {ret*100:.2f}%
- Volatilidade anual: {vol*100:.2f}%
- Sharpe Ratio: {sharpe:.3f}
- Max Drawdown: {max_dd*100:.2f}%
- Alfa (Jensen): {alfa*100:.2f}%
- Retorno acima/abaixo do mercado: {retorno_excedente*100:+.2f}%

**Classificações**
- Risco total: **{risco}**
- Horizonte mais adequado: **{horizonte}**
- Perfil do investidor: **{perfil}**
- Comparação com o mercado: **{comparacao}**

**Recomendação final:** **{rec}**
Confiança do score: **{conf*100:.1f}%**
Score quantitativo: **{score}**

**Leitura financeira**
"""

    if risco == "ALTO":
        interpretacao += (
            f"O ativo {ticker} apresenta risco elevado, com volatilidade e/ou drawdown relevantes. "
            "Pode entregar retornos interessantes, mas exige maior tolerância a oscilações."
        )
    elif risco == "MODERADO":
        interpretacao += (
            f"O ativo {ticker} apresenta risco moderado, com equilíbrio razoável entre retorno e oscilação. "
            "É mais compatível com investidores que aceitam alguma instabilidade."
        )
    else:
        interpretacao += (
            f"O ativo {ticker} apresenta risco relativamente baixo no período analisado, "
            "com comportamento mais estável do que ativos de risco alto."
        )

    if sharpe > 0.7:
        interpretacao += " A relação risco-retorno é boa, pois o Sharpe indica que o retorno compensa bem o risco."
    elif sharpe > 0:
        interpretacao += " A relação risco-retorno é aceitável, mas não excepcional."
    else:
        interpretacao += " O Sharpe negativo sugere que o retorno não compensou o risco assumido."

    if retorno_excedente > 0:
        interpretacao += " Em comparação com o mercado, o ativo apresentou desempenho superior no período analisado."
    elif retorno_excedente < 0:
        interpretacao += " Em comparação com o mercado, o ativo ficou abaixo do índice de referência no período analisado."
    else:
        interpretacao += " O ativo teve desempenho semelhante ao mercado no período analisado."

    if horizonte == "LONGO PRAZO":
        interpretacao += " Pela consistência observada, o ativo parece mais adequado para uma visão de longo prazo."
    elif horizonte == "MÉDIO PRAZO":
        interpretacao += " Os dados sugerem maior adequação para médio prazo."
    else:
        interpretacao += " A maior instabilidade indica que, no período analisado, ele se encaixa mais como oportunidade de curto prazo."

    return interpretacao


def analisar_graficos(metricas: dict) -> str:
    """Gera interpretação textual dos gráficos com base nos dados do ativo."""
    ticker = metricas["ticker"]
    precos = garantir_serie_1d(metricas["precos"], "preco")
    retornos = garantir_serie_1d(metricas["retorno_diario"], "retorno_diario")
    drawdown = garantir_serie_1d(metricas["dd_series"], "dd_series")

    tendencia = "alta" if precos.iloc[-1] > precos.iloc[0] else "baixa"

    vol = float(metricas["volatilidade_anual"])
    if vol > 0.35:
        vol_desc = "alta volatilidade"
    elif vol > 0.20:
        vol_desc = "volatilidade moderada"
    else:
        vol_desc = "baixa volatilidade"

    max_dd = float(metricas["max_drawdown"])
    if max_dd < -0.50:
        dd_desc = "quedas muito acentuadas"
    elif max_dd < -0.30:
        dd_desc = "quedas relevantes"
    else:
        dd_desc = "quedas mais controladas"

    skew = retornos.skew()
    if pd.isna(skew):
        dist_desc = "distribuição sem assimetria detectável"
    elif skew > 0:
        dist_desc = "maior presença de ganhos extremos do que de perdas extremas"
    else:
        dist_desc = "maior presença de perdas extremas do que de ganhos extremos"

    pior_queda = float(drawdown.min()) * 100

    texto = f"""
🤖 **INTERPRETAÇÃO DOS GRÁFICOS**

O ativo **{ticker}** apresenta tendência de **{tendencia}** ao longo do período analisado.

A análise de preço e retornos sugere **{vol_desc}**, o que indica oscilações relevantes no comportamento do ativo.

O gráfico de drawdown mostra **{dd_desc}**, com pior queda de aproximadamente **{pior_queda:.2f}%**.

A distribuição dos retornos sugere **{dist_desc}**.

Em conjunto, os gráficos indicam um ativo com comportamento compatível com o perfil de risco identificado nas métricas: o retorno pode ser interessante, mas as oscilações exigem maior tolerância ao risco.
"""
    return texto.strip()


def explicar_metricas() -> dict:
    """Textos curtos para cada métrica exibida."""
    return {
        "Retorno Anual": "Mostra quanto o ativo rendeu ao ano, em média. Quanto maior, melhor.",
        "Volatilidade Anual": "Mede o quanto o preço oscila. Quanto maior, maior o risco total.",
        "Retorno Mercado": "Retorno anual do índice de referência usado como comparação.",
        "Volatilidade Mercado": "Oscilação do mercado como um todo, servindo de base de risco.",
        "Retorno vs Mercado": "Diferença entre o retorno do ativo e o do mercado no período.",
        "Beta": "Sensibilidade do ativo em relação ao mercado. Acima de 1 tende a ser mais volátil que o índice.",
        "CAPM": "Retorno esperado teoricamente para o nível de risco assumido pelo ativo.",
        "Alfa (Jensen)": "Mostra se o ativo entregou retorno acima ou abaixo do esperado pelo CAPM.",
        "Sharpe Ratio": "Indica a relação entre retorno e risco total. Quanto maior, melhor.",
        "Sortino Ratio": "Semelhante ao Sharpe, mas considera só o risco de quedas.",
        "Max Drawdown": "Maior queda percentual do ativo no período analisado.",
        "Pior Drawdown": "Valor exato da maior perda observada.",
        "Data do Pior Drawdown": "Momento em que ocorreu a maior queda do ativo.",
        "Risco Total": "Classificação geral do risco com base em volatilidade, drawdown e beta.",
        "Horizonte Indicado": "Sinaliza se o ativo parece mais adequado para curto, médio ou longo prazo.",
        "Perfil do Investidor": "Tipo de investidor mais compatível com o ativo.",
        "Comparação com Mercado": "Mostra se o ativo teve desempenho superior, inferior ou igual ao índice.",
        "Período (dias)": "Quantidade de dias úteis ou pregão efetivamente usados na análise.",
    }


def processar_tickers_input(entrada: str) -> list:
    """Processa entrada de tickers (com quebra de linha ou vírgula)."""
    if "," in entrada:
        tickers = [t.strip().upper() for t in entrada.split(",")]
    else:
        tickers = [t.strip().upper() for t in entrada.split("\n")]
    
    return [t for t in tickers if t]


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.title("📊 Análise Financeira Multi-Ações + Markowitz")
st.markdown(
    "Análise de uma ou mais ações com métricas clássicas, comparação com mercado "
    "e otimização de portfólio usando Markowitz."
)

with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("📈 Ações para Análise")
    tickers_input = st.text_area(
        "Digite os tickers (um por linha ou separados por vírgula)",
        value="AMER3.SA",
        height=80
    )

    tickers_lista = processar_tickers_input(tickers_input)
    
    st.info(f"ℹ️ Total de ações: **{len(tickers_lista)}**")

    ticker_mercado = st.text_input("Ticker do Mercado (para comparação)", value="^BVSP").upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        data_inicio = st.text_input("Data Início (DD-MM-YYYY)", value="10-03-2023")
    with col2:
        data_fim = st.text_input("Data Fim (DD-MM-YYYY)", value="30-03-2026")

    botao_analisar = st.button("🚀 ANALISAR AGORA", use_container_width=True)

# Criar abas dinamicamente
if len(tickers_lista) == 1:
    tabs = st.tabs(["📈 Análise Completa", "🤖 IA & Recomendação", "📊 Gráficos"])
elif len(tickers_lista) > 1:
    tabs = st.tabs(["📈 Análise Completa", "🤖 IA & Recomendação", "📊 Gráficos", "📊 Comparativa", "🎯 Markowitz"])
else:
    tabs = st.tabs(["📈 Análise Completa", "🤖 IA & Recomendação", "📊 Gráficos"])


# ============================================================
# TAB 1: ANÁLISE COMPLETA
# ============================================================

with tabs[0]:
    if botao_analisar:
        with st.spinner("⏳ Baixando dados e calculando métricas..."):
            todas_metricas = {}
            for ticker in tickers_lista:
                metricas = calcular_metricas_completas(
                    ticker, ticker_mercado, data_inicio, data_fim
                )
                if metricas is not None:
                    todas_metricas[ticker] = metricas

        if not todas_metricas:
            st.error("❌ Erro ao baixar dados. Verifique tickers e período.")
        else:
            st.success(f"✅ Análise concluída para {len(todas_metricas)} ação(ões)")
            st.session_state.todas_metricas = todas_metricas

            for ticker, metricas in todas_metricas.items():
                with st.expander(f"📋 Detalhes de **{ticker}**", expanded=(len(todas_metricas) == 1)):
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("📈 Retorno Anual", f"{metricas['retorno_anual']*100:+.2f}%")

                    with col2:
                        st.metric("📊 Volatilidade", f"{metricas['volatilidade_anual']*100:.2f}%")

                    with col3:
                        st.metric("📉 Max Drawdown", f"{metricas['max_drawdown']*100:.2f}%")

                    with col4:
                        st.metric("🎯 Sharpe Ratio", f"{metricas['sharpe']:.3f}")

                    with col5:
                        st.metric("📌 Retorno vs Mercado", f"{metricas['retorno_excedente']*100:+.2f}%")

                    st.subheader("📋 Todas as Métricas")

                    risco_txt = classificar_risco(metricas)
                    horizonte_txt = classificar_horizonte(metricas)
                    perfil_txt = classificar_perfil_investidor(risco_txt)
                    comparacao_txt = comparar_com_mercado(metricas)

                    df_metricas = pd.DataFrame({
                        "Métrica": [
                            "Retorno Anual",
                            "Volatilidade Anual",
                            "Retorno Mercado",
                            "Volatilidade Mercado",
                            "Retorno vs Mercado",
                            "Beta",
                            "CAPM",
                            "Alfa (Jensen)",
                            "Sharpe Ratio",
                            "Sortino Ratio",
                            "Max Drawdown",
                            "Pior Drawdown",
                            "Data do Pior Drawdown",
                            "Risco Total",
                            "Horizonte Indicado",
                            "Perfil do Investidor",
                            "Comparação com Mercado",
                            "Período (dias)",
                        ],
                        "Valor": [
                            f"{metricas['retorno_anual']*100:+.2f}%",
                            f"{metricas['volatilidade_anual']*100:.2f}%",
                            f"{metricas['retorno_mercado']*100:+.2f}%",
                            f"{metricas['volatilidade_mercado']*100:.2f}%",
                            f"{metricas['retorno_excedente']*100:+.2f}%",
                            f"{metricas['beta']:.4f}",
                            f"{metricas['capm']*100:.2f}%",
                            f"{metricas['alfa']*100:+.2f}%",
                            f"{metricas['sharpe']:.4f}",
                            f"{metricas['sortino']:.4f}",
                            f"{metricas['max_drawdown']*100:.2f}%",
                            f"{metricas['pior_dd']*100:.2f}%",
                            metricas["pior_data_dd"].strftime("%d-%m-%Y") if hasattr(metricas["pior_data_dd"], "strftime") else str(metricas["pior_data_dd"]),
                            risco_txt,
                            horizonte_txt,
                            perfil_txt,
                            comparacao_txt,
                            f"{metricas['periodo_dias']}",
                        ]
                    })

                    st.dataframe(df_metricas, use_container_width=True)

                    st.divider()
                    st.subheader("🧾 Conclusão rápida")
                    if metricas["retorno_excedente"] > 0:
                        comparacao_final = "superou o mercado"
                    elif metricas["retorno_excedente"] < 0:
                        comparacao_final = "ficou abaixo do mercado"
                    else:
                        comparacao_final = "teve desempenho semelhante ao mercado"

                    st.write(
                        f"O ativo **{ticker}** apresentou risco **{risco_txt.lower()}**, "
                        f"com horizonte mais indicado de **{horizonte_txt.lower()}** e perfil mais compatível com investidor **{perfil_txt.lower()}**. "
                        f"No período analisado, o papel **{comparacao_final}**."
                    )

            st.divider()
            st.subheader("📚 Explicação das Métricas")

            with st.expander("📖 Clique para ver o significado de cada métrica"):
                explicacoes = explicar_metricas()
                for metrica, explicacao in explicacoes.items():
                    st.markdown(f"**{metrica}:** {explicacao}")

    else:
        st.info("ℹ️ Execute a análise na aba lateral para gerar as métricas.")


# ============================================================
# TAB 2: IA & RECOMENDAÇÃO
# ============================================================

with tabs[1]:
    if "todas_metricas" in st.session_state:
        todas_metricas = st.session_state.todas_metricas

        st.subheader("🤖 Análise automática de risco e retorno")
        st.markdown(
            "A recomendação principal é baseada em score quantitativo, com IA textual como apoio complementar."
        )

        with st.spinner("⏳ Carregando modelo de IA..."):
            modelo = carregar_modelo_classificacao()

        for ticker, metricas in todas_metricas.items():
            with st.expander(f"🤖 Recomendação para **{ticker}**", expanded=(len(todas_metricas) == 1)):
                resultado_ia = analisar_recomendacao_completa(modelo, metricas)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("💭 Risco Total")
                    risco = resultado_ia["risco"]
                    if risco == "ALTO":
                        st.error(f"❌ {risco}")
                    elif risco == "MODERADO":
                        st.warning(f"⚠️ {risco}")
                    else:
                        st.success(f"✅ {risco}")

                with col2:
                    st.subheader("📅 Horizonte")
                    horizonte = resultado_ia["horizonte"]
                    if horizonte == "LONGO PRAZO":
                        st.success(f"✅ {horizonte}")
                    elif horizonte == "MÉDIO PRAZO":
                        st.info(f"➡️ {horizonte}")
                    else:
                        st.warning(f"⚠️ {horizonte}")

                with col3:
                    st.subheader("🎯 Recomendação")
                    rec = resultado_ia["recomendacao"]
                    conf = resultado_ia["confianca"]
                    if rec == "COMPRAR":
                        st.success(f"✅ {rec} ({conf*100:.1f}%)")
                    elif rec == "VENDER":
                        st.error(f"❌ {rec} ({conf*100:.1f}%)")
                    else:
                        st.info(f"➡️ {rec} ({conf*100:.1f}%)")

                st.divider()

                col4, col5 = st.columns(2)

                with col4:
                    st.subheader("📌 Perfil do investidor")
                    perfil = resultado_ia["perfil"]
                    if perfil == "CONSERVADOR":
                        st.success(f"✅ {perfil}")
                    elif perfil == "MODERADO":
                        st.info(f"➡️ {perfil}")
                    else:
                        st.warning(f"⚠️ {perfil}")

                with col5:
                    st.subheader("📊 Comparação com mercado")
                    comp = resultado_ia["comparacao_mercado"]
                    if comp == "SUPEROU O MERCADO":
                        st.success(f"✅ {comp}")
                    elif comp == "FICOU ABAIXO DO MERCADO":
                        st.warning(f"⚠️ {comp}")
                    else:
                        st.info(f"➡️ {comp}")

                st.divider()
                st.markdown(gerar_interpretacao_ia(metricas, resultado_ia))

    else:
        st.info("ℹ️ Execute a análise na aba anterior primeiro.")


# ============================================================
# TAB 3: GRÁFICOS
# ============================================================

with tabs[2]:
    if "todas_metricas" in st.session_state:
        todas_metricas = st.session_state.todas_metricas

        for ticker, metricas in todas_metricas.items():
            with st.expander(f"📊 Gráficos de **{ticker}**", expanded=(len(todas_metricas) == 1)):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Evolução comparativa de R$100 investidos")

                    ativo_norm = garantir_serie_1d(
                        metricas["precos"] / metricas["precos"].iloc[0] * 100,
                        "Ativo"
                    )
                    mercado_norm = garantir_serie_1d(
                        metricas["precos_mercado"] / metricas["precos_mercado"].iloc[0] * 100,
                        "Mercado"
                    )

                    df_comp = pd.concat(
                        [ativo_norm, mercado_norm],
                        axis=1,
                        join="inner"
                    ).dropna()

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_comp.index, df_comp["Ativo"].values, label=metricas["ticker"], linewidth=2)
                    ax.plot(df_comp.index, df_comp["Mercado"].values, label=metricas["ticker_mercado"], linewidth=2)
                    ax.set_title("Crescimento de R$100 ao longo do período")
                    ax.set_xlabel("Data")
                    ax.set_ylabel("Valor acumulado (base 100)")
                    ax.grid(alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

                with col2:
                    st.subheader("📉 Maximum Drawdown")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    dd_values = np.asarray(garantir_serie_1d(metricas["dd_series"]).values).flatten() * 100
                    x_range = np.arange(len(dd_values))

                    ax.fill_between(x_range, dd_values, alpha=0.3)
                    ax.plot(x_range, dd_values, linewidth=1)
                    ax.set_title("Drawdown ao Longo do Tempo")
                    ax.set_xlabel("Dias")
                    ax.set_ylabel("Drawdown (%)")
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                st.subheader("📊 Distribuição de Retornos Diários")
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.hist(garantir_serie_1d(metricas["retorno_diario"]) * 100, bins=50, edgecolor="black", alpha=0.7)
                ax.set_title(f"Distribuição dos Retornos Diários - {metricas['ticker']}")
                ax.set_xlabel("Retorno Diário (%)")
                ax.set_ylabel("Frequência")
                ax.grid(alpha=0.3)
                st.pyplot(fig)

    else:
        st.info("ℹ️ Execute a análise na aba anterior primeiro.")


# ============================================================
# TAB 4: COMPARATIVA (Apenas se múltiplas ações)
# ============================================================

if len(tickers_lista) > 1:
    with tabs[3]:
        if "todas_metricas" in st.session_state:
            todas_metricas = st.session_state.todas_metricas

            st.subheader("📊 Análise Comparativa entre Ações")

            st.markdown("### 📋 Resumo Comparativo")

            df_comparacao = pd.DataFrame({
                "Ação": list(todas_metricas.keys()),
                "Retorno Anual (%)": [m["retorno_anual"] * 100 for m in todas_metricas.values()],
                "Volatilidade (%)": [m["volatilidade_anual"] * 100 for m in todas_metricas.values()],
                "Sharpe Ratio": [m["sharpe"] for m in todas_metricas.values()],
                "Max Drawdown (%)": [m["max_drawdown"] * 100 for m in todas_metricas.values()],
                "Retorno vs Mercado (%)": [m["retorno_excedente"] * 100 for m in todas_metricas.values()],
            })

            st.dataframe(df_comparacao, use_container_width=True, hide_index=True)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📈 Retorno vs Risco")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(
                    df_comparacao["Volatilidade (%)"],
                    df_comparacao["Retorno Anual (%)"],
                    s=300,
                    alpha=0.7,
                    c=df_comparacao["Sharpe Ratio"],
                    cmap="viridis"
                )

                for idx, row in df_comparacao.iterrows():
                    ax.annotate(
                        row["Ação"],
                        (row["Volatilidade (%)"], row["Retorno Anual (%)"]),
                        fontsize=11,
                        fontweight="bold",
                        ha="center"
                    )

                ax.set_xlabel("Volatilidade Anual (%)", fontsize=12)
                ax.set_ylabel("Retorno Anual (%)", fontsize=12)
                ax.set_title("Gráfico Retorno vs Risco", fontsize=14, fontweight="bold")
                ax.grid(alpha=0.3)
                st.pyplot(fig)

            with col2:
                st.markdown("### 📊 Evolução Normalizada")

                fig, ax = plt.subplots(figsize=(10, 6))

                for ticker, metricas in todas_metricas.items():
                    precos_norm = (metricas["precos"] / metricas["precos"].iloc[0]) * 100
                    ax.plot(precos_norm.index, precos_norm.values, label=ticker, linewidth=2)

                ax.set_xlabel("Data", fontsize=12)
                ax.set_ylabel("Valor de R$100 investido", fontsize=12)
                ax.set_title("Evolução Comparativa (R$100)", fontsize=14, fontweight="bold")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=10)
                st.pyplot(fig)

        else:
            st.info("ℹ️ Execute a análise na aba anterior primeiro.")

    # ============================================================
    # TAB 5: MARKOWITZ (Apenas se múltiplas ações)
    # ============================================================

    with tabs[4]:
        if "todas_metricas" in st.session_state:
            todas_metricas = st.session_state.todas_metricas

            st.subheader("🎯 Otimização de Portfólio - Teoria de Markowitz")
            st.markdown("Encontra a melhor alocação de portfólio baseada em retorno esperado e risco.")

            # Construir DataFrame com retornos alinhados
            df_retornos = construir_dataframe_retornos(todas_metricas)

            if df_retornos is not None and not df_retornos.empty:
                tickers_lista_optim = df_retornos.columns.tolist()

                # Calcular retornos anuais
                retornos_anuais = np.array([
                    anualizar_retorno(df_retornos[t].mean()) for t in tickers_lista_optim
                ])

                # Calcular matriz de covariância
                cov_matrix = calcular_matriz_covariancia(df_retornos).values

                # Otimizar Sharpe máximo
                st.markdown("### 🏆 Portfólio Ótimo (Máximo Sharpe Ratio)")

                resultado_otim = otimizar_sharpe_maximo(retornos_anuais, cov_matrix, tickers_lista_optim)

                if resultado_otim["sucesso"]:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("📈 Retorno Esperado", f"{resultado_otim['retorno']*100:.2f}%")

                    with col2:
                        st.metric("📊 Volatilidade", f"{resultado_otim['volatilidade']*100:.2f}%")

                    with col3:
                        st.metric("⭐ Sharpe Ratio", f"{resultado_otim['sharpe']:.4f}")

                    with col4:
                        st.metric("💼 Status", "✅ Otimizado")

                    st.divider()

                    st.subheader("💼 Alocação Recomendada")

                    df_alocacao = pd.DataFrame({
                        "Ação": tickers_lista_optim,
                        "Peso (%)": [w * 100 for w in resultado_otim['pesos']]
                    }).sort_values("Peso (%)", ascending=False)

                    st.dataframe(df_alocacao, use_container_width=True, hide_index=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.pie(
                            resultado_otim['pesos'],
                            labels=tickers_lista_optim,
                            autopct='%1.1f%%',
                            startangle=90
                        )
                        ax.set_title("Alocação Ótima de Portfólio")
                        st.pyplot(fig)

                    with col2:
                        st.markdown("""
                        #### 📊 Interpretação
                        
                        O portfólio acima é a **melhor combinação** das ações selecionadas, 
                        levando em conta:
                        
                        - **Retorno esperado**: Ganho anual estimado
                        - **Volatilidade**: Risco total da carteira
                        - **Correlação**: Como as ações se movem juntas
                        
                        Este portfólio maximiza o **Sharpe Ratio**, 
                        oferecendo o melhor retorno por unidade de risco.
                        """)

                    st.divider()

                    st.markdown("### 🔗 Análise de Correlações")

                    corr_matrix = calcular_matriz_correlacao(df_retornos)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(corr_matrix.columns)))
                    ax.set_yticks(range(len(corr_matrix.columns)))
                    ax.set_xticklabels(corr_matrix.columns)
                    ax.set_yticklabels(corr_matrix.columns)
                    plt.colorbar(im, ax=ax)
                    plt.title("Matriz de Correlação")
                    st.pyplot(fig)

                    st.markdown("""
                    **Dica:** Correlações próximas de 0 ou negativas indicam melhor diversificação.
                    """)

                else:
                    st.error("❌ Erro ao otimizar portfólio.")

            else:
                st.warning("⚠️ Sem dados suficientes para Markowitz.")

        else:
            st.info("ℹ️ Execute a análise na aba anterior primeiro.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p>🤖 IA: Hugging Face | 📊 Dados: Yahoo Finance | 🎯 Otimização: Scipy</p>
        <p>Análise financeira com suporte para múltiplas ações e Markowitz</p>
    </div>
    """,
    unsafe_allow_html=True
)
