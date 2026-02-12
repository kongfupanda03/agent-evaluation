"""
Database Initialization Module
Initializes ChromaDB for RAG and SQLite for NL2SQL with dummy data
"""

import os
import sqlite3
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


# Paths
DB_PATH = "/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/data/banking.db"
CHROMA_PATH = "/Users/xiongyuyu/Documents/projects/agent-eval/agent-evaluation/data/chroma_db"


def get_embeddings():
    """Initialize embeddings for vector store"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )


def init_sqlite_db():
    """Initialize SQLite database with dummy banking data"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create cap_pmi_summary_vw_tailored table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cap_pmi_summary_vw_tailored (
            client_name TEXT,
            product TEXT,
            revenue_ytd REAL,
            revenue_mtd REAL,
            total_assets REAL,
            total_liabilities REAL,
            contingent_exposure REAL,
            rwa REAL,
            income_ytd REAL,
            domicile_country TEXT,
            report_date TEXT
        )
    """)
    
    # Create cap_ccib_dealogic_summary_vw_tailored table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cap_ccib_dealogic_summary_vw_tailored (
            deal_id TEXT,
            client_name TEXT,
            product_type TEXT,
            deal_value_usd REAL,
            deal_status TEXT,
            scb_role TEXT,
            region TEXT,
            expected_close_date TEXT
        )
    """)
    
    # Create call_report_vw table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS call_report_vw (
            call_id TEXT,
            client_name TEXT,
            call_date TEXT,
            subject TEXT,
            objectives TEXT,
            meeting_notes TEXT,
            attendees TEXT,
            status TEXT
        )
    """)
    
    # Create consolidated_deal_pipeline_report_vw table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consolidated_deal_pipeline_report_vw (
            deal_id TEXT,
            client_name TEXT,
            project_name TEXT,
            deal_status TEXT,
            product_group TEXT,
            current_year_revenue REAL,
            annualized_revenue REAL,
            total_deal_size REAL,
            expected_revenue REAL,
            expected_close_date TEXT
        )
    """)
    
    # Insert dummy data - cap_pmi_summary
    clients = ["ABC Corp", "XYZ Inc", "DEF Ltd", "GHI Group", "JKL Enterprises", 
               "MNO Holdings", "PQR Industries", "STU Systems", "VWX Global", "YZA Partners"]
    products = ["Loans", "Trade Finance", "FX", "Cash Management", "Advisory"]
    
    dummy_pmi = []
    for i, client in enumerate(clients):
        for j, product in enumerate(products):
            dummy_pmi.append((
                client, product,
                round(1000000 + i * 500000 + j * 100000, 2),  # revenue_ytd
                round(80000 + i * 40000 + j * 8000, 2),       # revenue_mtd
                round(50000000 + i * 10000000, 2),            # total_assets
                round(20000000 + i * 5000000, 2),             # total_liabilities
                round(5000000 + i * 1000000, 2),              # contingent_exposure
                round(15000000 + i * 3000000, 2),             # rwa
                round(1200000 + i * 600000, 2),               # income_ytd
                ["Singapore", "Hong Kong", "UK", "USA", "Japan"][i % 5],
                "2025-01-31"
            ))
    
    cursor.executemany("""
        INSERT OR REPLACE INTO cap_pmi_summary_vw_tailored 
        (client_name, product, revenue_ytd, revenue_mtd, total_assets, total_liabilities, 
         contingent_exposure, rwa, income_ytd, domicile_country, report_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, dummy_pmi)
    
    # Insert dummy data - deals
    deal_statuses = ["Pending", "Approved", "Closed", "At Risk"]
    scb_roles = ["Lead", "Co-lead", "Participant"]
    
    dummy_deals = []
    for i in range(50):
        client = clients[i % len(clients)]
        dummy_deals.append((
            f"DEAL{1000 + i}",
            client,
            products[i % len(products)],
            round(10000000 + i * 2000000, 2),
            deal_statuses[i % len(deal_statuses)],
            scb_roles[i % len(scb_roles)],
            ["APAC", "EMEA", "Americas"][i % 3],
            f"2025-{((i % 12) + 1):02d}-15"
        ))
    
    cursor.executemany("""
        INSERT OR REPLACE INTO cap_ccib_dealogic_summary_vw_tailored
        (deal_id, client_name, product_type, deal_value_usd, deal_status, scb_role, region, expected_close_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, dummy_deals)
    
    # Insert dummy data - call reports
    dummy_calls = []
    for i in range(30):
        client = clients[i % len(clients)]
        dummy_calls.append((
            f"CALL{2000 + i}",
            client,
            f"2025-{((i % 12) + 1):02d}-{((i % 28) + 1):02d}",
            ["Quarterly Review", "Deal Discussion", "Credit Update", "Relationship Meeting"][i % 4],
            ["Review performance", "Discuss new deal", "Update on facilities", "Strengthen relationship"][i % 4],
            f"Meeting held with {client}. Discussed ongoing business and future opportunities.",
            f"Client: {client}, Bank: Relationship Manager, Credit Analyst",
            "Completed"
        ))
    
    cursor.executemany("""
        INSERT OR REPLACE INTO call_report_vw
        (call_id, client_name, call_date, subject, objectives, meeting_notes, attendees, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, dummy_calls)
    
    # Insert dummy data - pipeline
    dummy_pipeline = []
    for i in range(40):
        client = clients[i % len(clients)]
        dummy_pipeline.append((
            f"PIPE{3000 + i}",
            client,
            f"Project {client} Expansion",
            ["Prospecting", "Negotiation", "Documentation", "Closing"][i % 4],
            products[i % len(products)],
            round(500000 + i * 100000, 2),
            round(2000000 + i * 400000, 2),
            round(10000000 + i * 2000000, 2),
            round(800000 + i * 150000, 2),
            f"2025-{((i % 12) + 1):02d}-28"
        ))
    
    cursor.executemany("""
        INSERT OR REPLACE INTO consolidated_deal_pipeline_report_vw
        (deal_id, client_name, project_name, deal_status, product_group, current_year_revenue,
         annualized_revenue, total_deal_size, expected_revenue, expected_close_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, dummy_pipeline)
    
    conn.commit()
    conn.close()
    print(f"SQLite database initialized at: {DB_PATH}")


def init_chroma_db():
    """Initialize ChromaDB with dummy documents for RAG"""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    embeddings = get_embeddings()
    
    # Dummy documents for RAG
    documents = [
        # ESG Reports
        Document(
            page_content="ABC Corp has committed to achieving net-zero carbon emissions by 2040. Their sustainability strategy focuses on renewable energy adoption, supply chain decarbonization, and green financing initiatives. The company has already reduced emissions by 25% since 2020.",
            metadata={"source": "ABC Corp ESG Report 2024", "client": "ABC Corp", "doc_type": "ESG Report"}
        ),
        Document(
            page_content="XYZ Inc's ESG commitments include 100% renewable energy by 2035, water conservation targets of 30% reduction, and diversity goals of 40% women in leadership positions. Their sustainability score improved from B+ to A- this year.",
            metadata={"source": "XYZ Inc Sustainability Report", "client": "XYZ Inc", "doc_type": "ESG Report"}
        ),
        Document(
            page_content="DEF Ltd has established comprehensive ESG targets including carbon neutrality by 2035, zero waste to landfill by 2028, and sustainable sourcing for 90% of materials. They have been recognized as an ESG leader in their sector.",
            metadata={"source": "DEF Ltd Annual Report", "client": "DEF Ltd", "doc_type": "ESG Report"}
        ),
        # Credit Assessments
        Document(
            page_content="Credit Assessment for ABC Corp: Rating AA-. Strong financial position with stable cash flows. Low leverage at 2.1x EBITDA. Diversified revenue streams across 15 countries. Risk factors include exposure to commodity price volatility.",
            metadata={"source": "ABC Corp Credit Assessment", "client": "ABC Corp", "doc_type": "Credit Assessment"}
        ),
        Document(
            page_content="Credit Assessment for XYZ Inc: Rating A+. Solid balance sheet with conservative leverage of 1.8x. Strong market position in technology sector. Key risks include customer concentration and regulatory changes in data privacy.",
            metadata={"source": "XYZ Inc Credit Rating", "client": "XYZ Inc", "doc_type": "Credit Assessment"}
        ),
        Document(
            page_content="Credit Assessment for GHI Group: Rating BBB+. Moderate leverage at 3.2x. Improving trend in debt reduction. Exposure to cyclical industries requires monitoring. Adequate liquidity with $500M cash reserves.",
            metadata={"source": "GHI Group Credit Review", "client": "GHI Group", "doc_type": "Credit Assessment"}
        ),
        # Deal Documentation
        Document(
            page_content="Deal Structure for Project ABC Expansion: Total facility size $500M, split between Term Loan ($300M) and Revolver ($200M). Tenor: 5 years. Pricing: SOFR + 150bps. Security: Corporate guarantee. Conditions precedent include board approval and legal opinions.",
            metadata={"source": "ABC Corp Deal Documentation", "client": "ABC Corp", "doc_type": "Deal Structure"}
        ),
        Document(
            page_content="Approval Conditions for XYZ Inc Facility: 1) Completion of environmental due diligence, 2) Execution of intercreditor agreement, 3) Minimum liquidity covenant of $100M, 4) Quarterly financial reporting within 45 days.",
            metadata={"source": "XYZ Inc Approval Letter", "client": "XYZ Inc", "doc_type": "Approval Conditions"}
        ),
        # Governance
        Document(
            page_content="ABC Corp Board of Directors: Chairman - John Smith (Independent), CEO - Jane Doe, CFO - Bob Johnson, Independent Directors - Sarah Lee, Mike Chen, Lisa Wong. Board meets quarterly with strong attendance record.",
            metadata={"source": "ABC Corp Governance Report", "client": "ABC Corp", "doc_type": "Governance"}
        ),
        Document(
            page_content="XYZ Inc Senior Management: CEO - Michael Brown (appointed 2022), CFO - Emily Davis, COO - David Wilson. Executive team has average tenure of 8 years. Succession planning in place for all key positions.",
            metadata={"source": "XYZ Inc Governance Disclosure", "client": "XYZ Inc", "doc_type": "Governance"}
        ),
        # Annual Reports
        Document(
            page_content="ABC Corp Annual Report Highlights: Revenue $5.2B (+12% YoY), Net Income $800M (+15%), Total Assets $50B. Strategic priorities include digital transformation and international expansion into Southeast Asia.",
            metadata={"source": "ABC Corp Annual Report 2024", "client": "ABC Corp", "doc_type": "Annual Report"}
        ),
        Document(
            page_content="XYZ Inc Financial Summary: Revenue $3.8B (+8%), EBITDA $950M (+10%), Free Cash Flow $600M. Strong performance driven by cloud services growth. Headquartered in Singapore with operations in 12 countries.",
            metadata={"source": "XYZ Inc Annual Report", "client": "XYZ Inc", "doc_type": "Annual Report"}
        ),
        Document(
            page_content="DEF Ltd Business Overview: Leading manufacturer with 15,000 employees globally. Revenue $2.1B from three business segments: Industrial (45%), Consumer (35%), and Services (20%). Headquarters located in London, UK.",
            metadata={"source": "DEF Ltd Annual Report", "client": "DEF Ltd", "doc_type": "Annual Report"}
        ),
        # Risk Analysis
        Document(
            page_content="Risk Analysis for JKL Enterprises: Key risks include currency exposure (60% revenue in USD), supply chain concentration (top 3 suppliers = 45%), and regulatory changes in EU markets. Mitigation strategies include hedging and supplier diversification.",
            metadata={"source": "JKL Risk Assessment", "client": "JKL Enterprises", "doc_type": "Risk Analysis"}
        ),
        Document(
            page_content="MNO Holdings Risk Profile: Moderate risk rating. Exposure to real estate cycle. Geographic concentration in Hong Kong and mainland China. Liquidity risk managed through committed credit lines totaling $2B.",
            metadata={"source": "MNO Risk Analysis", "client": "MNO Holdings", "doc_type": "Risk Analysis"}
        ),
    ]
    
    # Create or load Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="banking_documents"
    )
    
    print(f"ChromaDB initialized at: {CHROMA_PATH} with {len(documents)} documents")
    return vectorstore


def init_all_databases():
    """Initialize both ChromaDB and SQLite databases"""
    print("Initializing databases...")
    init_sqlite_db()
    init_chroma_db()
    print("Database initialization complete!")


def get_vectorstore():
    """Get or create Chroma vector store"""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="banking_documents"
    )


def get_db_connection():
    """Get SQLite database connection"""
    return sqlite3.connect(DB_PATH)


def get_database_schema() -> str:
    """Get database schema for SQL generation context"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    schema_info = []
    
    tables = [
        "cap_pmi_summary_vw_tailored",
        "cap_ccib_dealogic_summary_vw_tailored",
        "call_report_vw",
        "consolidated_deal_pipeline_report_vw"
    ]
    
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        column_info = [f"  - {col[1]} ({col[2]})" for col in columns]
        schema_info.append(f"Table: {table}\n" + "\n".join(column_info))
    
    conn.close()
    return "\n\n".join(schema_info)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize databases for the agent")
    parser.add_argument("--sqlite", action="store_true", help="Initialize only SQLite")
    parser.add_argument("--chroma", action="store_true", help="Initialize only ChromaDB")
    
    args = parser.parse_args()
    
    if args.sqlite:
        init_sqlite_db()
    elif args.chroma:
        init_chroma_db()
    else:
        init_all_databases()
