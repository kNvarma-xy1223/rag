"""
Generate mock CSV and PDF datasets for testing the RAG system.
Run: python data/generate_mock_data.py
"""
import csv
import os
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent


# ─────────────────────────────────────────────
# CSV 1: English sales data (numerical)
# ─────────────────────────────────────────────

def generate_sales_csv():
    regions = ["North", "South", "East", "West", "Central"]
    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Service Pro", "Service Lite"]
    quarters = ["Q1-2023", "Q2-2023", "Q3-2023", "Q4-2023", "Q1-2024", "Q2-2024"]

    rows = []
    for i in range(120):
        region = random.choice(regions)
        product = random.choice(products)
        quarter = random.choice(quarters)
        units = random.randint(50, 5000)
        unit_price = round(random.uniform(9.99, 499.99), 2)
        discount = round(random.uniform(0.0, 0.25), 4)
        revenue = round(units * unit_price * (1 - discount), 2)
        cogs = round(revenue * random.uniform(0.30, 0.60), 2)
        gross_profit = round(revenue - cogs, 2)
        margin_pct = round((gross_profit / revenue) * 100, 2) if revenue else 0.0
        customer_count = random.randint(5, 200)
        nps_score = round(random.uniform(20.0, 90.0), 1)

        rows.append({
            "transaction_id": f"TXN-{i+1:04d}",
            "quarter": quarter,
            "region": region,
            "product": product,
            "units_sold": units,
            "unit_price_usd": unit_price,
            "discount_rate": discount,
            "revenue_usd": revenue,
            "cogs_usd": cogs,
            "gross_profit_usd": gross_profit,
            "gross_margin_pct": margin_pct,
            "customer_count": customer_count,
            "nps_score": nps_score,
        })

    path = DATA_DIR / "sales_data.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created: {path} ({len(rows)} rows)")


# ─────────────────────────────────────────────
# CSV 2: Spanish financial data
# ─────────────────────────────────────────────

def generate_ventas_csv():
    regiones = ["Norte", "Sur", "Este", "Oeste", "Centro"]
    productos = ["Producto A", "Producto B", "Servicio X", "Servicio Y"]
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
             "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

    rows = []
    for i in range(80):
        region = random.choice(regiones)
        producto = random.choice(productos)
        mes = random.choice(meses)
        unidades = random.randint(20, 3000)
        precio_unitario = round(random.uniform(5.0, 250.0), 2)
        descuento = round(random.uniform(0.0, 0.20), 4)
        ingresos = round(unidades * precio_unitario * (1 - descuento), 2)
        costos = round(ingresos * random.uniform(0.35, 0.55), 2)
        ganancia = round(ingresos - costos, 2)
        margen_pct = round((ganancia / ingresos) * 100, 2) if ingresos else 0.0
        clientes = random.randint(3, 150)
        satisfaccion = round(random.uniform(3.0, 5.0), 1)

        rows.append({
            "id_transaccion": f"VTA-{i+1:04d}",
            "mes": mes,
            "region": region,
            "producto": producto,
            "unidades_vendidas": unidades,
            "precio_unitario_eur": precio_unitario,
            "tasa_descuento": descuento,
            "ingresos_eur": ingresos,
            "costos_eur": costos,
            "ganancia_bruta_eur": ganancia,
            "margen_bruto_pct": margen_pct,
            "num_clientes": clientes,
            "satisfaccion_cliente": satisfaccion,
        })

    path = DATA_DIR / "ventas_datos.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created: {path} ({len(rows)} rows)")


# ─────────────────────────────────────────────
# PDF generation with reportlab
# ─────────────────────────────────────────────

def generate_pdfs():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_LEFT
    except ImportError:
        print("reportlab not installed. Skipping PDF generation.")
        print("Install with: pip install reportlab")
        return

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=13, spaceAfter=6)

    # ── English PDF ──────────────────────────────────────────────
    en_path = DATA_DIR / "corporate_report_en.pdf"
    doc = SimpleDocTemplate(str(en_path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []

    en_content = [
        ("Acme Corporation — Q2 2024 Quarterly Report",
         "Executive Summary\n"
         "Acme Corporation delivered strong financial results in Q2 2024, with total consolidated revenue "
         "reaching $47.3 million, representing a 12.4% increase year-over-year from $42.1 million in Q2 2023. "
         "Gross profit margin improved by 2.1 percentage points to 58.7%, driven by operational efficiencies "
         "and favorable product mix shifts toward higher-margin software services."),
        ("Revenue Breakdown by Segment",
         "The North America segment contributed $28.6 million (60.5% of total), up 9.8% from the prior year. "
         "The EMEA segment generated $11.2 million (23.7%), showing robust growth of 18.3% year-over-year. "
         "Asia-Pacific revenues reached $7.5 million (15.8%), a 10.2% increase driven by expansion in "
         "Southeast Asian markets. Total recurring subscription revenue was $31.4 million, representing "
         "66.4% of total revenue, compared to 61.2% in the same period last year."),
        ("Cost Structure and Profitability",
         "Cost of goods sold decreased as a percentage of revenue to 41.3% from 43.4% a year ago. "
         "Research and development expenses were $6.8 million (14.4% of revenue), up 22.1% as the company "
         "accelerates investment in artificial intelligence product features. Sales and marketing expense "
         "was $9.2 million (19.5% of revenue). General and administrative costs were $4.1 million. "
         "EBITDA for the quarter was $9.6 million, yielding an EBITDA margin of 20.3%."),
        ("Customer Metrics",
         "Total active customers at quarter-end reached 8,740, an increase of 1,230 net new customers "
         "compared to Q1 2024. Annual Recurring Revenue (ARR) expanded to $127.4 million from $112.8 million "
         "at the end of Q1 2024, representing 13.0% sequential growth. Net Revenue Retention Rate (NRR) "
         "was 118%, indicating strong expansion revenue from the existing customer base. "
         "Customer Acquisition Cost (CAC) improved to $1,840 from $2,110 in Q1 2024."),
        ("Balance Sheet and Cash Flow",
         "Cash and cash equivalents at quarter-end were $38.2 million. Operating cash flow was $11.4 million, "
         "with free cash flow of $9.1 million after $2.3 million in capital expenditures. "
         "Total debt outstanding is $15.0 million under the revolving credit facility at an interest rate of "
         "SOFR plus 1.75%, currently approximately 7.1% per annum. Accounts receivable days outstanding "
         "(DSO) improved to 38 days from 44 days in the prior quarter."),
        ("Guidance and Outlook",
         "For Q3 2024, management expects revenue in the range of $49.0 million to $50.5 million, "
         "representing growth of 11% to 14% year-over-year. Full-year 2024 revenue guidance is raised "
         "to $193 million to $197 million, from the prior range of $188 million to $193 million. "
         "Full-year EBITDA margin is expected to be in the range of 19% to 21%."),
        ("Risk Factors",
         "The company faces macroeconomic headwinds including potential enterprise budget contractions. "
         "Foreign exchange fluctuations could impact EMEA and APAC revenues, with every 1% change in "
         "the EUR/USD rate affecting annual revenue by approximately $0.8 million. Competition from "
         "larger technology vendors continues to intensify in core product categories."),
    ]

    for title, content in en_content:
        story.append(Paragraph(title, heading))
        story.append(Paragraph(content.replace("\n", " "), body))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    print(f"Created: {en_path}")

    # ── Spanish PDF ──────────────────────────────────────────────
    es_path = DATA_DIR / "informe_corporativo_es.pdf"
    doc2 = SimpleDocTemplate(str(es_path), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story2 = []

    es_content = [
        ("Corporación Acme — Informe Trimestral Q2 2024",
         "Resumen Ejecutivo\n"
         "La Corporación Acme logró sólidos resultados financieros en el segundo trimestre de 2024, "
         "con ingresos consolidados totales de 47,3 millones de dólares, lo que representa un incremento "
         "del 12,4% interanual respecto a los 42,1 millones del Q2 2023. El margen de ganancia bruta "
         "mejoró 2,1 puntos porcentuales hasta alcanzar el 58,7%, impulsado por eficiencias operativas "
         "y un cambio favorable en la mezcla de productos hacia servicios de software de mayor margen."),
        ("Desglose de Ingresos por Segmento",
         "El segmento de América del Norte aportó 28,6 millones de dólares (60,5% del total), un 9,8% "
         "más que el año anterior. El segmento EMEA generó 11,2 millones (23,7%), con un sólido crecimiento "
         "del 18,3% interanual. Los ingresos de Asia-Pacífico alcanzaron los 7,5 millones de dólares (15,8%), "
         "un incremento del 10,2% impulsado por la expansión en los mercados del Sudeste Asiático. "
         "Los ingresos recurrentes por suscripción fueron de 31,4 millones, representando el 66,4% "
         "de los ingresos totales, frente al 61,2% del mismo período del año anterior."),
        ("Estructura de Costos y Rentabilidad",
         "El costo de ventas disminuyó como porcentaje de los ingresos al 41,3% desde el 43,4% del año "
         "anterior. Los gastos en investigación y desarrollo fueron de 6,8 millones de dólares (14,4% de los "
         "ingresos), un 22,1% más, ya que la empresa acelera las inversiones en funcionalidades de "
         "inteligencia artificial. Los gastos de ventas y marketing ascendieron a 9,2 millones (19,5% "
         "de los ingresos). El EBITDA del trimestre fue de 9,6 millones, con un margen EBITDA del 20,3%."),
        ("Métricas de Clientes",
         "Los clientes activos al cierre del trimestre alcanzaron los 8.740, un incremento de 1.230 "
         "nuevos clientes netos respecto al Q1 2024. Los Ingresos Recurrentes Anuales (ARR) se expandieron "
         "a 127,4 millones de dólares desde los 112,8 millones al cierre del Q1 2024, representando un "
         "crecimiento secuencial del 13,0%. La Tasa de Retención Neta de Ingresos (NRR) fue del 118%. "
         "El Costo de Adquisición de Clientes (CAC) mejoró a 1.840 dólares desde los 2.110 del Q1 2024."),
        ("Balance y Flujo de Caja",
         "El efectivo y equivalentes al cierre del trimestre fueron de 38,2 millones de dólares. "
         "El flujo de caja operativo fue de 11,4 millones, con un flujo de caja libre de 9,1 millones "
         "tras 2,3 millones en gastos de capital. La deuda total pendiente es de 15,0 millones de dólares "
         "bajo la línea de crédito revolvente a SOFR más 1,75%, actualmente aproximadamente el 7,1% anual. "
         "Los días de cuentas por cobrar (DSO) mejoraron a 38 días desde los 44 días del trimestre anterior."),
        ("Perspectivas y Guía",
         "Para el Q3 2024, la dirección espera ingresos en el rango de 49,0 a 50,5 millones de dólares, "
         "lo que representa un crecimiento del 11% al 14% interanual. La guía de ingresos para todo el "
         "año 2024 se eleva a entre 193 y 197 millones de dólares, desde el rango anterior de 188 a "
         "193 millones. El margen EBITDA para todo el año se espera que esté en el rango del 19% al 21%."),
        ("Análisis de Riesgos",
         "La empresa enfrenta vientos en contra macroeconómicos, incluidas posibles contracciones en "
         "los presupuestos empresariales. Las fluctuaciones del tipo de cambio podrían impactar los ingresos "
         "de EMEA y APAC; cada cambio del 1% en el tipo EUR/USD afecta los ingresos anuales en "
         "aproximadamente 0,8 millones de dólares. La competencia de proveedores tecnológicos más "
         "grandes continúa intensificándose en las categorías de productos principales."),
    ]

    for title, content in es_content:
        story2.append(Paragraph(title, heading))
        story2.append(Paragraph(content.replace("\n", " "), body))
        story2.append(Spacer(1, 0.3*cm))

    doc2.build(story2)
    print(f"Created: {es_path}")


if __name__ == "__main__":
    print("Generating mock datasets...")
    generate_sales_csv()
    generate_ventas_csv()
    generate_pdfs()
    print("Done. Files written to ./data/")
