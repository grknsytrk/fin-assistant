from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_config


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    from src.ingest import ingest_raw_pdfs

    pages, summary = ingest_raw_pdfs(
        raw_dir=Path(args.raw_dir),
        output_file=Path(args.output_file),
    )
    print("PDF dosyalari bulundu:")
    for name in summary["pdf_files"]:
        print(f"- {name}")
    print(f"Toplam PDF: {summary['num_pdfs']}")
    print("PDF basina sayfa:")
    for name, page_count in summary["pages_per_pdf"].items():
        print(f"- {name}: {page_count}")
    print(f"Toplam sayfa: {summary['total_pages']}")
    if summary.get("companies"):
        print(f"Sirketler: {', '.join(summary['companies'])}")
    print(f"Kaydedilen sayfa kaydi: {len(pages)}")
    print(f"Cikti dosyasi: {args.output_file}")


def cmd_index(args: argparse.Namespace) -> None:
    from src.index import build_index

    summary = build_index(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("Index ozeti")
    print(f"PDF sayisi: {summary['num_pdfs']}")
    print("Bulunan PDF dosyalari:")
    for name in summary["pdf_files"]:
        print(f"- {name}")
    print("PDF basina sayfa sayisi:")
    for name, page_count in summary["pages_per_pdf"].items():
        print(f"- {name}: {page_count}")
    print(f"Toplam sayfa: {summary['total_pages']}")
    print(f"Toplam chunk: {summary['total_chunks']}")
    print(f"Indexlenen chunk: {summary['indexed_chunks']}")
    print(f"Collection count: {summary['collection_count']}")
    print(f"Index basarisi: {summary['indexing_success']}")
    print(f"Collection: {summary['collection_name']}")
    print(f"Chroma path: {summary['chroma_path']}")


def cmd_index_v2(args: argparse.Namespace) -> None:
    from src.index import build_index_v2

    summary = build_index_v2(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("Index v2 ozeti")
    print(f"PDF sayisi: {summary['num_pdfs']}")
    print("Bulunan PDF dosyalari:")
    for name in summary["pdf_files"]:
        print(f"- {name}")
    print("PDF basina sayfa sayisi:")
    for name, page_count in summary["pages_per_pdf"].items():
        print(f"- {name}: {page_count}")
    print(f"Toplam sayfa: {summary['total_pages']}")
    print(f"Toplam chunk (v2): {summary['total_chunks']}")
    print(f"Indexlenen chunk (v2): {summary['indexed_chunks']}")
    print(f"Collection count: {summary['collection_count']}")
    print(f"Index basarisi: {summary['indexing_success']}")
    print(f"Collection: {summary['collection_name']}")
    print(f"Chroma path: {summary['chroma_path']}")
    print(f"Sections detected: {summary['sections_detected']}")
    print(f"Table-like chunk count: {summary['table_like_count']}")
    print("Chunks per PDF:")
    for doc_id, chunk_count in summary["chunks_per_pdf"].items():
        print(f"- {doc_id}: {chunk_count}")


def cmd_index_v2_incremental(args: argparse.Namespace) -> None:
    from src.index import build_index_v2_incremental

    summary = build_index_v2_incremental(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("Index v2 incremental ozeti")
    print(f"PDF sayisi: {summary['num_pdfs']}")
    print(f"Toplam sayfa: {summary['total_pages']}")
    print(f"Toplam chunk (v2): {summary['total_chunks']}")
    print(f"Bu kosuda indexlenen chunk (v2): {summary['indexed_chunks']}")
    print(f"Collection count: {summary['collection_count']}")
    print(f"Index basarisi: {summary['indexing_success']}")
    print(f"Collection: {summary['collection_name']}")
    print(f"Chroma path: {summary['chroma_path']}")
    print("Degisen dosyalar:")
    for name in summary.get("changed_files", []):
        print(f"- {name}")
    if summary.get("removed_files"):
        print("Silinen dosyalar:")
        for name in summary.get("removed_files", []):
            print(f"- {name}")


def cmd_query(args: argparse.Namespace) -> None:
    from src.retrieve import Retriever

    retriever = Retriever(
        chroma_path=Path(args.chroma_path),
        collection_name=args.collection_name,
        model_name=args.model_name,
    )
    rows = retriever.retrieve(args.query, top_k=args.top_k, company=args.company)
    if not rows:
        print("Sonuc bulunamadi.")
        return

    company_info = f" | company={args.company}" if args.company else ""
    print(f"Sorgu: {args.query}{company_info}")
    print(f"Top {len(rows)} sonuc:")
    for idx, row in enumerate(rows, start=1):
        preview = row.text[:400].replace("\n", " ")
        print(
            f"{idx}. distance={row.distance:.4f} score={row.score:.4f} "
            f"doc_id={row.doc_id} company={row.company} quarter={row.quarter} page={row.page} chunk_id={row.chunk_id}"
        )
        print(f"   text={preview}")


def cmd_query_v2(args: argparse.Namespace) -> None:
    from src.retrieve import RetrieverV2

    retriever = RetrieverV2(
        chroma_path=Path(args.chroma_path),
        collection_name=args.collection_name,
        model_name=args.model_name,
    )
    rows = retriever.retrieve_with_boost(
        query=args.query,
        top_k_initial=args.top_k_initial,
        top_k_final=args.top_k,
        alpha=args.alpha,
        quarter=args.quarter,
        company=args.company,
    )
    if not rows:
        print("Sonuc bulunamadi.")
        return

    quarter_info = f" | quarter_filter={args.quarter}" if args.quarter else ""
    company_info = f" | company={args.company}" if args.company else ""
    print(f"Sorgu (v2): {args.query}{quarter_info}{company_info}")
    print(f"Top {len(rows)} sonuc (rerank):")
    for idx, row in enumerate(rows, start=1):
        preview = row.text[:400].replace("\n", " ")
        print(
            f"{idx}. distance={row.distance:.4f} raw_score={row.score:.4f} "
            f"vector_score={row.vector_score:.4f} lexical_boost={row.lexical_boost:.2f} "
            f"final_score={row.final_score:.4f}"
        )
        print(
            f"   doc_id={row.doc_id} company={row.company} quarter={row.quarter} page={row.page} "
            f"chunk_id={row.chunk_id} section_title={row.section_title} block_type={row.block_type}"
        )
        print(f"   text={preview}")


def cmd_query_v3(args: argparse.Namespace) -> None:
    from src.query_parser import parse_query
    from src.retrieve import RetrieverV3

    parsed = parse_query(args.query)
    auto_quarter = parsed.get("quarter")
    query_type = parsed.get("signals", {}).get("query_type", "qualitative")

    retriever = RetrieverV3(
        chroma_path=Path(args.chroma_path),
        collection_name=args.collection_name,
        model_name=args.model_name,
    )
    rows = retriever.retrieve_with_query_awareness(
        query=args.query,
        top_k_initial=args.top_k_initial,
        top_k_final=args.top_k,
        alpha=args.alpha,
        company_override=args.company,
    )
    if not rows:
        print("Sonuc bulunamadi.")
        return

    company_info = f" | company={args.company}" if args.company else ""
    print(
        f"Sorgu (v3): {args.query} | auto_quarter={auto_quarter} | "
        f"query_type={query_type}{company_info}"
    )
    print(f"Top {len(rows)} sonuc (query-aware rerank):")
    for idx, row in enumerate(rows, start=1):
        preview = row.text[:400].replace("\n", " ")
        print(
            f"{idx}. distance={row.distance:.4f} raw_score={row.score:.4f} "
            f"vector_score={row.vector_score:.4f} lexical_boost={row.lexical_boost:.2f} "
            f"final_score={row.final_score:.4f}"
        )
        print(
            f"   doc_id={row.doc_id} company={row.company} quarter={row.quarter} page={row.page} "
            f"chunk_id={row.chunk_id} section_title={row.section_title} block_type={row.block_type}"
        )
        print(f"   text={preview}")


def cmd_ask(args: argparse.Namespace) -> None:
    from src.answer import AnswerEngine, RulesBasedAnswerAdapter
    from src.retrieve import Retriever

    retriever = Retriever(
        chroma_path=Path(args.chroma_path),
        collection_name=args.collection_name,
        model_name=args.model_name,
    )
    chunks = retriever.retrieve(args.question, top_k=args.top_k, company=args.company)
    engine = AnswerEngine(adapter=RulesBasedAnswerAdapter(max_distance=args.max_distance))
    answer = engine.answer(args.question, chunks)
    print(answer)


def cmd_ask_v3(args: argparse.Namespace) -> None:
    from src.answer import AnswerEngine, RulesBasedAnswerAdapter
    from src.retrieve import RetrieverV3

    retriever = RetrieverV3(
        chroma_path=Path(args.chroma_path),
        collection_name=args.collection_name,
        model_name=args.model_name,
    )
    chunks = retriever.retrieve_with_query_awareness(
        query=args.question,
        top_k_initial=args.top_k_initial,
        top_k_final=args.top_k,
        alpha=args.alpha,
        company_override=args.company,
    )
    engine = AnswerEngine(adapter=RulesBasedAnswerAdapter(max_distance=args.max_distance))
    answer = engine.answer(args.question, chunks)
    print(answer)


def cmd_eval(args: argparse.Namespace) -> None:
    from src.eval_runner import run_retrieval_eval

    summary = run_retrieval_eval(
        questions_file=Path(args.questions_file),
        output_file=Path(args.output_file),
        top_k=args.top_k,
    )
    print(f"Soru sayisi: {summary['questions_count']}")
    print(f"Top-k: {summary['top_k']}")
    print(f"Cikti: {summary['output_file']}")


def cmd_eval_compare(args: argparse.Namespace) -> None:
    from src.eval_runner import run_retrieval_eval_comparison

    summary = run_retrieval_eval_comparison(
        questions_file=Path(args.questions_file),
        output_file=Path(args.output_file),
        top_k=args.top_k,
        top_k_initial_v2=args.top_k_initial_v2,
        alpha=args.alpha,
    )
    print(f"Soru sayisi: {summary['questions_count']}")
    print(f"Top-k (final): {summary['top_k']}")
    print(f"Top-k initial (v2): {summary['top_k_initial_v2']}")
    print(f"Alpha: {summary['alpha']}")
    print(f"Cikti: {summary['output_file']}")


def cmd_metrics_report(args: argparse.Namespace) -> None:
    from src.metrics import run_metrics_report

    report = run_metrics_report(
        gold_file=Path(args.gold_file),
        multi_company_gold_file=Path(args.multi_company_gold_file),
        detailed_output=Path(args.detailed_output),
        summary_output=Path(args.summary_output),
        week6_summary_output=Path(args.week6_summary_output),
        top_k=args.top_k,
        top_k_initial_v2=args.top_k_initial_v2,
        top_k_initial_v3=args.top_k_initial_v3,
        top_k_initial_v5_vector=args.top_k_initial_v5_vector,
        top_k_initial_v5_bm25=args.top_k_initial_v5_bm25,
        top_k_candidates_v6=args.top_k_candidates_v6,
        alpha_v2=args.alpha_v2,
        alpha_v3=args.alpha_v3,
        beta_v5=args.beta_v5,
    )
    print("Metrics report")
    print(report["table"])
    multi = report["summary"].get("multi_company_extraction", {})
    by_company = multi.get("by_company", {})
    if by_company:
        print("Multi-company extraction accuracy (before -> after)")
        for company, row in by_company.items():
            print(
                f"- {company}: before={row.get('extraction_accuracy_before', row.get('coverage_rate_before', 0.0)):.4f} "
                f"after={row.get('extraction_accuracy_after', row.get('coverage_rate_after', row.get('coverage_rate', 0.0))):.4f} "
                f"delta={row.get('extraction_accuracy_delta', 0.0):+.4f} "
                f"invalid={row.get('invalid_rate', 0.0):.4f} "
                f"verified_pass={row.get('verified_pass_rate', 0.0):.4f} "
                f"count={row.get('count', 0)}"
            )
    print(f"Summary json: {report['summary_output']}")
    print(f"Week6 summary json: {report['week6_summary_output']}")
    print(f"Detailed jsonl: {report['detailed_output']}")


def cmd_benchmark_week6(args: argparse.Namespace) -> None:
    from src.metrics import run_metrics_report

    report = run_metrics_report(
        gold_file=Path(args.gold_file),
        multi_company_gold_file=Path(args.multi_company_gold_file),
        detailed_output=Path(args.detailed_output),
        summary_output=Path(args.summary_output),
        week6_summary_output=Path(args.week6_summary_output),
        top_k=args.top_k,
        top_k_initial_v2=args.top_k_initial_v2,
        top_k_initial_v3=args.top_k_initial_v3,
        top_k_initial_v5_vector=args.top_k_initial_v5_vector,
        top_k_initial_v5_bm25=args.top_k_initial_v5_bm25,
        top_k_candidates_v6=args.top_k_candidates_v6,
        alpha_v2=args.alpha_v2,
        alpha_v3=args.alpha_v3,
        beta_v5=args.beta_v5,
    )
    print("Week6 benchmark report")
    print(report["table"])
    multi = report["summary"].get("multi_company_extraction", {})
    by_company = multi.get("by_company", {})
    if by_company:
        print("Multi-company extraction accuracy (before -> after)")
        for company, row in by_company.items():
            print(
                f"- {company}: before={row.get('extraction_accuracy_before', row.get('coverage_rate_before', 0.0)):.4f} "
                f"after={row.get('extraction_accuracy_after', row.get('coverage_rate_after', row.get('coverage_rate', 0.0))):.4f} "
                f"delta={row.get('extraction_accuracy_delta', 0.0):+.4f} "
                f"invalid={row.get('invalid_rate', 0.0):.4f} "
                f"verified_pass={row.get('verified_pass_rate', 0.0):.4f} "
                f"count={row.get('count', 0)}"
            )
    print(f"Summary json: {report['summary_output']}")
    print(f"Week6 summary json: {report['week6_summary_output']}")
    print(f"Detailed jsonl: {report['detailed_output']}")


def cmd_error_report(args: argparse.Namespace) -> None:
    from src.error_analysis import run_error_report

    retrievers = [item.strip() for item in args.retrievers.split(",") if item.strip()] if args.retrievers else None
    report = run_error_report(
        detailed_file=Path(args.detailed_file),
        output_file=Path(args.output_file),
        retrievers=retrievers,
    )
    print("Error analysis report")
    print(f"Detailed file: {report['detailed_file']}")
    print(f"Output file: {report['output_file']}")
    print(f"Questions count: {report['questions_count']}")
    print(f"Total errors: {report['total_errors']}")
    print("Errors per retriever:")
    for retriever_name, count in report["errors_per_retriever"].items():
        print(f"- {retriever_name}: {count}")


def cmd_latency_bench(args: argparse.Namespace) -> None:
    from src.latency_benchmark import run_latency_benchmark

    report = run_latency_benchmark(
        gold_file=Path(args.gold_file),
        output_file=Path(args.output_file),
        sample_size=args.sample_size,
        top_k=args.top_k,
        top_k_initial_v3=args.top_k_initial_v3,
        top_k_initial_v5_vector=args.top_k_initial_v5_vector,
        top_k_initial_v5_bm25=args.top_k_initial_v5_bm25,
        top_k_candidates_v6=args.top_k_candidates_v6,
        alpha_v3=args.alpha_v3,
        beta_v5=args.beta_v5,
    )
    print("Latency benchmark")
    print(report["table"])
    print(f"Output json: {report['output_file']}")


def cmd_coverage_audit(args: argparse.Namespace) -> None:
    from src.coverage_audit import run_coverage_audit

    report = run_coverage_audit(
        company=args.company,
        gold_file=Path(args.gold_file),
        output_file=Path(args.output_file) if args.output_file else None,
        top_k_initial_v3=args.top_k_initial_v3,
        top_k=args.top_k,
        alpha_v3=args.alpha_v3,
    )
    print("Coverage audit")
    print(f"Company: {report.get('company')}")
    print(f"Questions: {report.get('num_questions')}")
    print("Coverage by metric")
    for metric, row in report.get("coverage_by_metric", {}).items():
        print(
            f"- {metric}: coverage={row.get('coverage_rate', 0.0):.4f} "
            f"invalid={row.get('invalid_rate', 0.0):.4f} "
            f"verified_pass={row.get('verified_pass_rate', 0.0):.4f}"
        )
    print("Top missing metrics")
    for row in report.get("top_missing_metrics", [])[:5]:
        print(
            f"- {row.get('metric')}: missing={row.get('missing_count')} "
            f"coverage={row.get('coverage_rate', 0.0):.4f} reasons={row.get('likely_reasons', {})}"
        )
    print(f"Output json: {report.get('output_file')}")


def cmd_dict_suggest(args: argparse.Namespace) -> None:
    from src.coverage_audit import suggest_dictionary_phrases

    report = suggest_dictionary_phrases(
        company=args.company,
        gold_file=Path(args.gold_file),
        top_n=args.top_n,
        top_k_initial_v3=args.top_k_initial_v3,
        top_k=args.top_k,
        alpha_v3=args.alpha_v3,
    )
    print("Dictionary phrase suggestions")
    print(f"Company: {report.get('company')}")
    print(f"Top {report.get('top_n')} phrases:")
    for item in report.get("suggestions", []):
        print(f"- {item.get('phrase')}: {item.get('count')}")


def cmd_doctor(args: argparse.Namespace) -> None:
    from src.doctor import _print_table, run_doctor

    results = run_doctor(Path(args.config))
    _print_table(results)


def build_parser(config_path: Path | None = None) -> argparse.ArgumentParser:
    cfg = load_config(config_path or Path("config.yaml"))
    parser = argparse.ArgumentParser(description="RAG-Fin v1 CLI")
    parser.add_argument(
        "--config",
        default=str(config_path or Path("config.yaml")),
        help="Config dosyasi yolu (ornek: config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="PDF'leri sayfa bazli cikar")
    ingest_parser.add_argument("--raw-dir", default=str(cfg.paths.raw_dir))
    ingest_parser.add_argument("--output-file", default=str(cfg.paths.pages_file))
    ingest_parser.set_defaults(func=cmd_ingest)

    index_parser = sub.add_parser("index", help="Ingest + chunk + embed + index")
    index_parser.add_argument("--raw-dir", default=str(cfg.paths.raw_dir))
    index_parser.add_argument("--processed-dir", default=str(cfg.paths.processed_dir))
    index_parser.add_argument("--collection-name", default=cfg.chroma.collection_v1)
    index_parser.add_argument("--chunk-size", type=int, default=cfg.chunking.v1.chunk_size)
    index_parser.add_argument("--overlap", type=int, default=cfg.chunking.v1.overlap)
    index_parser.set_defaults(func=cmd_index)

    index_v2_parser = sub.add_parser("index_v2", help="Week-2: heading-aware chunk + rerank index")
    index_v2_parser.add_argument("--raw-dir", default=str(cfg.paths.raw_dir))
    index_v2_parser.add_argument("--processed-dir", default=str(cfg.paths.processed_dir))
    index_v2_parser.add_argument("--collection-name", default=cfg.chroma.collection_v2)
    index_v2_parser.add_argument("--chunk-size", type=int, default=cfg.chunking.v2.chunk_size)
    index_v2_parser.add_argument("--overlap", type=int, default=cfg.chunking.v2.overlap)
    index_v2_parser.set_defaults(func=cmd_index_v2)

    index_v2_inc_parser = sub.add_parser(
        "index_v2_incremental",
        help="Week-17: sadece yeni/degisen PDFleri indexle (v2)",
    )
    index_v2_inc_parser.add_argument("--raw-dir", default=str(cfg.paths.raw_dir))
    index_v2_inc_parser.add_argument("--processed-dir", default=str(cfg.paths.processed_dir))
    index_v2_inc_parser.add_argument("--collection-name", default=cfg.chroma.collection_v2)
    index_v2_inc_parser.add_argument("--chunk-size", type=int, default=cfg.chunking.v2.chunk_size)
    index_v2_inc_parser.add_argument("--overlap", type=int, default=cfg.chunking.v2.overlap)
    index_v2_inc_parser.set_defaults(func=cmd_index_v2_incremental)

    query_parser = sub.add_parser("query", help="Top-k retrieval")
    query_parser.add_argument("query")
    query_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    query_parser.add_argument("--collection-name", default=cfg.chroma.collection_v1)
    query_parser.add_argument("--chroma-path", default=str(cfg.chroma.dir))
    query_parser.add_argument("--model-name", default=cfg.models.embedding)
    query_parser.add_argument("--company", default=None, help="Sirket filtresi (orn: BIM)")
    query_parser.set_defaults(func=cmd_query)

    query_v2_parser = sub.add_parser("query_v2", help="Week-2 retrieval + lexical rerank")
    query_v2_parser.add_argument("query")
    query_v2_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    query_v2_parser.add_argument("--top-k-initial", type=int, default=cfg.retrieval.v2_top_k_initial)
    query_v2_parser.add_argument("--alpha", type=float, default=cfg.retrieval.alpha_v2)
    query_v2_parser.add_argument("--quarter", default=None)
    query_v2_parser.add_argument("--company", default=None, help="Sirket filtresi (orn: BIM)")
    query_v2_parser.add_argument("--collection-name", default=cfg.chroma.collection_v2)
    query_v2_parser.add_argument("--chroma-path", default=str(cfg.chroma.dir))
    query_v2_parser.add_argument("--model-name", default=cfg.models.embedding)
    query_v2_parser.set_defaults(func=cmd_query_v2)

    query_v3_parser = sub.add_parser("query_v3", help="Week-3 query parser + type-aware retrieval")
    query_v3_parser.add_argument("query")
    query_v3_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    query_v3_parser.add_argument("--top-k-initial", type=int, default=cfg.retrieval.v3_top_k_initial)
    query_v3_parser.add_argument("--alpha", type=float, default=cfg.retrieval.alpha_v3)
    query_v3_parser.add_argument("--company", default=None, help="Sirket filtresi (orn: BIM)")
    query_v3_parser.add_argument("--collection-name", default=cfg.chroma.collection_v2)
    query_v3_parser.add_argument("--chroma-path", default=str(cfg.chroma.dir))
    query_v3_parser.add_argument("--model-name", default=cfg.models.embedding)
    query_v3_parser.set_defaults(func=cmd_query_v3)

    ask_parser = sub.add_parser("ask", help="Grounded answer")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    ask_parser.add_argument("--max-distance", type=float, default=0.45)
    ask_parser.add_argument("--company", default=None, help="Sirket filtresi (orn: BIM)")
    ask_parser.add_argument("--collection-name", default=cfg.chroma.collection_v1)
    ask_parser.add_argument("--chroma-path", default=str(cfg.chroma.dir))
    ask_parser.add_argument("--model-name", default=cfg.models.embedding)
    ask_parser.set_defaults(func=cmd_ask)

    ask_v3_parser = sub.add_parser("ask_v3", help="Week-3 grounded answer with query-aware retrieval")
    ask_v3_parser.add_argument("question")
    ask_v3_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    ask_v3_parser.add_argument("--top-k-initial", type=int, default=cfg.retrieval.v3_top_k_initial)
    ask_v3_parser.add_argument("--alpha", type=float, default=cfg.retrieval.alpha_v3)
    ask_v3_parser.add_argument("--max-distance", type=float, default=0.45)
    ask_v3_parser.add_argument("--company", default=None, help="Sirket filtresi (orn: BIM)")
    ask_v3_parser.add_argument("--collection-name", default=cfg.chroma.collection_v2)
    ask_v3_parser.add_argument("--chroma-path", default=str(cfg.chroma.dir))
    ask_v3_parser.add_argument("--model-name", default=cfg.models.embedding)
    ask_v3_parser.set_defaults(func=cmd_ask_v3)

    eval_parser = sub.add_parser("eval", help="questions.jsonl icin retrieval eval")
    eval_parser.add_argument("--questions-file", default="eval/questions.jsonl")
    eval_parser.add_argument("--output-file", default=str(cfg.paths.processed_dir / "eval_retrieval.jsonl"))
    eval_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    eval_parser.set_defaults(func=cmd_eval)

    eval_compare_parser = sub.add_parser(
        "eval_compare",
        help="Week-2: v1-v2 retrieval comparison eval",
    )
    eval_compare_parser.add_argument("--questions-file", default="eval/questions.jsonl")
    eval_compare_parser.add_argument(
        "--output-file",
        default=str(cfg.paths.processed_dir / "eval_retrieval_comparison.jsonl"),
    )
    eval_compare_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    eval_compare_parser.add_argument("--top-k-initial-v2", type=int, default=cfg.retrieval.v2_top_k_initial)
    eval_compare_parser.add_argument("--alpha", type=float, default=cfg.retrieval.alpha_v2)
    eval_compare_parser.set_defaults(func=cmd_eval_compare)

    metrics_parser = sub.add_parser("metrics_report", help="Week-6 retrieval metrics report (v1..v6)")
    metrics_parser.add_argument("--gold-file", default=str(cfg.evaluation.gold_file))
    metrics_parser.add_argument("--multi-company-gold-file", default=str(cfg.evaluation.gold_multicompany_file))
    metrics_parser.add_argument("--detailed-output", default=str(cfg.evaluation.detailed_output))
    metrics_parser.add_argument("--summary-output", default=str(cfg.evaluation.summary_output))
    metrics_parser.add_argument("--week6-summary-output", default=str(cfg.evaluation.week6_summary_output))
    metrics_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    metrics_parser.add_argument("--top-k-initial-v2", type=int, default=cfg.retrieval.v2_top_k_initial)
    metrics_parser.add_argument("--top-k-initial-v3", type=int, default=cfg.retrieval.v3_top_k_initial)
    metrics_parser.add_argument("--top-k-initial-v5-vector", type=int, default=cfg.retrieval.v5_top_k_vector)
    metrics_parser.add_argument("--top-k-initial-v5-bm25", type=int, default=cfg.retrieval.v5_top_k_bm25)
    metrics_parser.add_argument("--top-k-candidates-v6", type=int, default=cfg.retrieval.v6_cross_top_n)
    metrics_parser.add_argument("--alpha-v2", type=float, default=cfg.retrieval.alpha_v2)
    metrics_parser.add_argument("--alpha-v3", type=float, default=cfg.retrieval.alpha_v3)
    metrics_parser.add_argument("--beta-v5", type=float, default=cfg.retrieval.beta_v5)
    metrics_parser.set_defaults(func=cmd_metrics_report)

    benchmark_week6_parser = sub.add_parser(
        "benchmark_week6",
        help="Week-6 benchmark: v1/v2/v3/v4_bm25/v5_hybrid/v6_cross",
    )
    benchmark_week6_parser.add_argument("--gold-file", default=str(cfg.evaluation.gold_file))
    benchmark_week6_parser.add_argument("--multi-company-gold-file", default=str(cfg.evaluation.gold_multicompany_file))
    benchmark_week6_parser.add_argument("--detailed-output", default=str(cfg.evaluation.detailed_output))
    benchmark_week6_parser.add_argument("--summary-output", default=str(cfg.evaluation.summary_output))
    benchmark_week6_parser.add_argument("--week6-summary-output", default=str(cfg.evaluation.week6_summary_output))
    benchmark_week6_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    benchmark_week6_parser.add_argument("--top-k-initial-v2", type=int, default=cfg.retrieval.v2_top_k_initial)
    benchmark_week6_parser.add_argument("--top-k-initial-v3", type=int, default=cfg.retrieval.v3_top_k_initial)
    benchmark_week6_parser.add_argument("--top-k-initial-v5-vector", type=int, default=cfg.retrieval.v5_top_k_vector)
    benchmark_week6_parser.add_argument("--top-k-initial-v5-bm25", type=int, default=cfg.retrieval.v5_top_k_bm25)
    benchmark_week6_parser.add_argument("--top-k-candidates-v6", type=int, default=cfg.retrieval.v6_cross_top_n)
    benchmark_week6_parser.add_argument("--alpha-v2", type=float, default=cfg.retrieval.alpha_v2)
    benchmark_week6_parser.add_argument("--alpha-v3", type=float, default=cfg.retrieval.alpha_v3)
    benchmark_week6_parser.add_argument("--beta-v5", type=float, default=cfg.retrieval.beta_v5)
    benchmark_week6_parser.set_defaults(func=cmd_benchmark_week6)

    error_report_parser = sub.add_parser(
        "error_report",
        help="Week-6 failed retrieval error analysis raporu",
    )
    error_report_parser.add_argument("--detailed-file", default=str(cfg.evaluation.detailed_output))
    error_report_parser.add_argument("--output-file", default=str(cfg.evaluation.error_output))
    error_report_parser.add_argument(
        "--retrievers",
        default=None,
        help="Opsiyonel: v1,v2,v3,v4_bm25,v5_hybrid,v6_cross",
    )
    error_report_parser.set_defaults(func=cmd_error_report)

    latency_parser = sub.add_parser("latency_bench", help="Week-7 latency benchmark (v3/v5/v6)")
    latency_parser.add_argument("--gold-file", default=str(cfg.evaluation.gold_file))
    latency_parser.add_argument("--output-file", default=str(cfg.evaluation.latency_output))
    latency_parser.add_argument("--sample-size", type=int, default=cfg.evaluation.latency_sample_size)
    latency_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    latency_parser.add_argument("--top-k-initial-v3", type=int, default=cfg.retrieval.v3_top_k_initial)
    latency_parser.add_argument("--top-k-initial-v5-vector", type=int, default=cfg.retrieval.v5_top_k_vector)
    latency_parser.add_argument("--top-k-initial-v5-bm25", type=int, default=cfg.retrieval.v5_top_k_bm25)
    latency_parser.add_argument("--top-k-candidates-v6", type=int, default=cfg.retrieval.v6_cross_top_n)
    latency_parser.add_argument("--alpha-v3", type=float, default=cfg.retrieval.alpha_v3)
    latency_parser.add_argument("--beta-v5", type=float, default=cfg.retrieval.beta_v5)
    latency_parser.set_defaults(func=cmd_latency_bench)

    coverage_parser = sub.add_parser(
        "coverage_audit",
        help="Week-12 coverage diagnostics (metric coverage + missing reasons)",
    )
    coverage_parser.add_argument("--company", default=None, help="Sirket kodu (orn: MIGROS, SOK, BIM)")
    coverage_parser.add_argument("--gold-file", default=str(cfg.evaluation.gold_multicompany_file))
    coverage_parser.add_argument("--output-file", default=None)
    coverage_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    coverage_parser.add_argument("--top-k-initial-v3", type=int, default=cfg.retrieval.v3_top_k_initial)
    coverage_parser.add_argument("--alpha-v3", type=float, default=cfg.retrieval.alpha_v3)
    coverage_parser.set_defaults(func=cmd_coverage_audit)

    dict_suggest_parser = sub.add_parser(
        "dict_suggest",
        help="Week-12 dictionary expansion helper (missing-case phrase mining)",
    )
    dict_suggest_parser.add_argument("--company", required=True, help="Sirket kodu (orn: SOK)")
    dict_suggest_parser.add_argument("--gold-file", default=str(cfg.evaluation.gold_multicompany_file))
    dict_suggest_parser.add_argument("--top-n", type=int, default=30)
    dict_suggest_parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    dict_suggest_parser.add_argument("--top-k-initial-v3", type=int, default=cfg.retrieval.v3_top_k_initial)
    dict_suggest_parser.add_argument("--alpha-v3", type=float, default=cfg.retrieval.alpha_v3)
    dict_suggest_parser.set_defaults(func=cmd_dict_suggest)

    doctor_parser = sub.add_parser(
        "doctor",
        help="Ortam ve bagimlilik kontrolleri (python/chromadb/model/path/chroma)",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    return parser


def main() -> None:
    setup_logging()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="config.yaml")
    pre_args, remaining = pre_parser.parse_known_args()

    parser = build_parser(Path(pre_args.config))
    args = parser.parse_args(remaining)
    args.func(args)


if __name__ == "__main__":
    main()
