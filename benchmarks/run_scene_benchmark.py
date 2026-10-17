"""Scene benchmark: full image -> PaddleDetector -> crop -> recognizer -> regions.
End-to-end measured separately from recognition-only. Stdlib + cv2/numpy.
Usage: python benchmarks/run_scene_benchmark.py --data benchmark_data --recognizer paddle_v5_latin
"""
from __future__ import annotations
import argparse, csv, json, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from benchmarks.dataset import count_records, filter_records, load_annotations, build_manifest, real_reviewed_test_count
from importlib.metadata import version, PackageNotFoundError
from benchmarks.metrics import accent_stripped_cer, cer, detection_metrics, diacritic_loss_rate, exact_match, polygon_iou, wer, field_exact_match
from app.services.polygon_utils import crop_polygon
ROOT = Path(__file__).resolve().parents[1]
def _git_hash():
    try:
        import subprocess as sp
        r=sp.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True,timeout=3)
        return r.stdout.strip() if r.returncode==0 else None
    except Exception: return None
def _load_annotations(data_dir: Path):
    p=data_dir/"annotations.jsonl" if (data_dir/"annotations.jsonl").exists() else data_dir
    if p.is_dir(): p=p/"annotations.jsonl"
    return load_annotations(p)
def _apply_preprocessing(crop, mode: str):
    if crop is None: return crop
    if mode=="original": return crop
    if mode=="gray":
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    if mode=="otsu":
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        _,bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cv2.cvtColor(bw,cv2.COLOR_GRAY2BGR)
    return crop
def _get_detector():
    from app.detectors.paddle_detector import PaddleDetector
    return PaddleDetector()
def _get_recognizer(name: str):
    k=(name or "auto").lower()
    if k in ("tesseract","tesseract_vie_eng"):
        from app.recognizers.tesseract_recognizer import TesseractRecognizer
        return TesseractRecognizer(),"tesseract","vie+eng"
    from app.recognizers.paddle_recognizer import PaddleRecognizer
    return PaddleRecognizer(),"paddle_v5_latin","latin_PP-OCRv5_mobile_rec"
def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="benchmark_data")
    ap.add_argument("--recognizer",default="auto",choices=["paddle_v5_latin","paddle","tesseract","auto"])
    ap.add_argument('--mode', default='production', choices=['production', 'ablation'])
    ap.add_argument('--route', default='scene', choices=['scene', 'document', 'auto'])
    ap.add_argument('--iou-backend', default='opencv', choices=['opencv', 'bbox', 'shapely'])
    ap.add_argument("--preprocessing",default="original",choices=["original","gray","otsu"])
    ap.add_argument("--psm", type=int, default=None, help="Document/legacy PSM; hybrid line crops use production policy")
    ap.add_argument("--iou-threshold",type=float,default=0.5)
    ap.add_argument("--output",default="benchmark_results")
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--source",default=None, choices=["real","synthetic"])
    ap.add_argument("--split",default=None, choices=["train","val","test"])
    ap.add_argument("--reviewed-only",action="store_true")
    ap.add_argument("--include-unreviewed",action="store_true")
    ap.add_argument("--include-rejected",action="store_true")
    args=ap.parse_args()
    production = args.mode == 'production'
    if production and args.preprocessing != 'original':
        ap.error('Preprocessing ablation requires --mode ablation')
    args.psm = args.psm if args.psm is not None else (6 if production else 7)
    from functools import partial
    score_iou = partial(polygon_iou, backend=args.iou_backend)
    data_dir=Path(args.data)
    if not data_dir.is_absolute():
        alt=ROOT/args.data
        if alt.exists(): data_dir=alt.resolve()
        else: data_dir=data_dir.resolve()
    rows_ann=_load_annotations(data_dir)
    if not rows_ann:
        print(f"No annotations in {data_dir}",file=sys.stderr); return 1
    rows_ann=filter_records(rows_ann, split=args.split, source=args.source, reviewed_only=args.reviewed_only, include_unreviewed=args.include_unreviewed, include_rejected=args.include_rejected)
    if args.limit: rows_ann=rows_ann[:args.limit]
    if not rows_ann:
        print("No records after filtering", file=sys.stderr); return 1
    bench_root=data_dir if data_dir.is_dir() else data_dir.parent
    manifest=build_manifest(rows_ann, bench_root)
    if production:
        from app.pipelines.router import get_router
        pipeline=get_router()
        rec_label, rec_model = args.recognizer, 'resolved_by_production_router'
    else:
        detector=_get_detector(); rec,rec_label,rec_model=_get_recognizer(args.recognizer)
    languages=sorted({a.get("language", "vie+eng") for a in rows_ann})
    if rec_label == "tesseract":
        rec_model=" / ".join({"vi":"vie", "en":"eng"}.get(lang,lang) for lang in languages)
    import sys as _sys
    py_ver=_sys.version.split()[0]
    try:
        paddle_ver=version('paddlepaddle')
    except PackageNotFoundError: paddle_ver="not_installed"
    try:
        po_ver=version('paddleocr')
    except PackageNotFoundError: po_ver="not_installed"
    try:
        import pytesseract as _pt; tesseract_ver=str(_pt.get_tesseract_version())
    except Exception: tesseract_ver="not_available"
    from app.core.config import get_settings
    from benchmarks.dataset import model_fingerprints
    from app.services.paddle_models import resolve_rec_model
    settings=get_settings()
    device=settings.device
    det_model="PP-OCRv5_server_det"
    # model load timing cold
    t0=time.perf_counter()
    det_ready=False; rec_ready=False
    try:
        if production: det_ready=True
        elif hasattr(detector,"ensure_ready"): det_ready=detector.ensure_ready()
        else: det_ready=detector.status()=="ready"
    except Exception: det_ready=False
    try:
        if production: rec_ready=True
        elif hasattr(rec,"ensure_ready"): rec_ready=all(rec.ensure_ready(lang=lang) for lang in languages)
        else: rec_ready=rec.status()=="ready"
    except Exception: rec_ready=False
    model_load_ms=round((time.perf_counter()-t0)*1000,2)
    if not det_ready or not rec_ready:
        print("Required OCR components unavailable", file=sys.stderr); return 1
    ts=datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_id=f"{ts}_scene_{rec_label}_{args.preprocessing}"
    out_dir=ROOT/args.output/run_id
    out_dir.mkdir(parents=True,exist_ok=False)
    # aggregate
    scene_rows=[]; rec_rows=[]; errors=[]
    total_gt=0; total_pred=0; total_matched=0; total_missed=0; total_fp=0
    cer_list=[]; wer_list=[]; accent_list=[]; loss_list=[]; exact_cnt=0
    det_times=[]; rec_times=[]; total_times=[]
    conf_bins={"0.0-0.5":[0,0,0.0],"0.5-0.7":[0,0,0.0],"0.7-0.8":[0,0,0.0],"0.8-0.9":[0,0,0.0],"0.9-1.0":[0,0,0.0]}
    first_det_ms=None; first_rec_ms=None
    page_cers=[]; page_wers=[]; page_exacts=[]
    successes=0; actual_pipelines=[]; empty_predictions=0
    field_total=0; field_ok=0; field_unmapped=0
    for idx,ann in enumerate(rows_ann):
        img_rel=ann.get("image","")
        img_path=(bench_root/img_rel) if not Path(img_rel).is_absolute() else Path(img_rel)
        if not img_path.exists():
            alt2=ROOT/img_rel
            if alt2.exists(): img_path=alt2
        image=cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot decode benchmark image: {img_path}")
        gt_regions=[r for r in ann.get("regions",[]) if not r.get("ignore", False)]
        ignored=[r["polygon"] for r in ann.get("regions",[]) if r.get("ignore", False)]
        gt_polys=[r["polygon"] for r in gt_regions]
        gt_texts=[str(r.get("text","")) for r in gt_regions]
        fields=ann.get('fields') or {}
        mapped={r['field_key'] for r in gt_regions if r.get('field_key')}
        field_unmapped+=len(set(fields)-mapped)
        total_gt+=len(gt_polys)
        # detection timing (baseline original image, no preprocessing)
        t_det0=time.perf_counter()
        pipeline_result=None
        try:
            if production:
                pipeline_result=pipeline.process(img_path.read_bytes(), route=args.route, recognizer=args.recognizer,
                                                 lang=ann.get('language', 'vie+eng'), psm=args.psm, filename=img_path.name)
                detections=pipeline_result.regions
                actual_pipelines.append(pipeline_result.provenance.to_dict())
            else:
                detections=sorted(detector.detect(image), key=lambda d: (d.bbox.y, d.bbox.x))
            detections=[d for d in detections if not any(score_iou(d.polygon, p)>=args.iou_threshold for p in ignored)
                        or any(score_iou(d.polygon, p)>=args.iou_threshold for p in gt_polys)]
            pred_polys=[d.polygon for d in detections]
            warnings = getattr(pipeline_result, 'warnings', []) if production else []
            all_regions_failed = production and bool(detections) and all(not d.raw_text for d in detections) and any(
                'recognition failed' in w or 'recognition error' in w for w in warnings)
            successes+=int(not all_regions_failed)
        except Exception as e:
            pred_polys=[]; pred_scores=[]; detections=[]
            errors.append({"id":ann.get("id"),"error":str(e),"type":"detection_failed"})
        det_ms=round((time.perf_counter()-t_det0)*1000,2)
        if production and pipeline_result is not None:
            det_ms=pipeline_result.timing_ms.detection or 0
        det_times.append(det_ms)
        if idx==0: first_det_ms=det_ms
        total_pred+=len(pred_polys)
        # detection metrics per image
        dm=detection_metrics(pred_polys, gt_polys, iou_threshold=args.iou_threshold, backend=args.iou_backend)
        total_matched+=dm["matched"]; total_missed+=dm["missed"]; total_fp+=dm["fp"]
        # recognition per detected region -> match to GT via IoU greedy to score CER/WER
        t_rec0=time.perf_counter()
        # need mapping pred idx -> gt idx via greedy like detection_metrics
        # build per-region pred texts
        pred_texts=[]; pred_scores=[]; recognition_failures=0
        for d_poly in pred_polys:
            if production:
                region=detections[len(pred_texts)]
                pred_texts.append(region.raw_text or '')
                pred_scores.append(region.ocr_confidence)
                continue
            crop=crop_polygon(image, d_poly)
            if crop is None or crop.size==0:
                recognition_failures+=1
                pred_texts.append(""); pred_scores.append(None); continue
            crop_p=_apply_preprocessing(crop, args.preprocessing)
            try:
                language=ann.get("language","vie+eng")
                if rec_label == "tesseract":
                    language={"vi":"vie", "en":"eng"}.get(language, language)
                res=rec.recognize(crop_p, lang=language, psm=args.psm)
                pred_texts.append(res.raw_text or "")
                pred_scores.append(res.ocr_confidence)
            except Exception as e:
                recognition_failures+=1
                pred_texts.append(""); pred_scores.append(None)
                errors.append({"id":ann.get("id"),"error":str(e),"type":"recognition_failed"})
        if not production and pred_polys and recognition_failures == len(pred_polys):
            successes-=1
        rec_ms=round((time.perf_counter()-t_rec0)*1000,2)
        if production and pipeline_result is not None:
            rec_ms=pipeline_result.timing_ms.recognition or 0
        rec_times.append(rec_ms)
        if idx==0: first_rec_ms=rec_ms
        total_ms=(pipeline_result.timing_ms.total or 0) if production and pipeline_result is not None else det_ms+rec_ms
        total_times.append(total_ms)
        # align via IoU greedy: for each gt find best pred
        used=set()
        for gt_idx,gt_poly in enumerate(gt_polys):
            best_i,best_v=-1,0.0
            for i,p_poly in enumerate(pred_polys):
                if i in used: continue
                v=score_iou(gt_poly,p_poly)
                if v>best_v: best_v,best_i=v,i
            matched = best_v>=args.iou_threshold and best_i!=-1
            if matched:
                used.add(best_i)
            pred=pred_texts[best_i] if matched else ""
            gt=gt_texts[gt_idx]
            field_key=gt_regions[gt_idx].get('field_key')
            if field_key:
                field_total+=1
                field_ok+=int(field_exact_match(pred,fields[field_key]))
            c=cer(pred,gt); w=wer(pred,gt); ac=accent_stripped_cer(pred,gt); loss=diacritic_loss_rate(pred,gt); ex=exact_match(pred,gt)
            cer_list.append(c); wer_list.append(w); accent_list.append(ac); loss_list.append(loss)
            if ex: exact_cnt+=1
            conf=pred_scores[best_i] if matched else None
            if conf is not None:
                k="0.0-0.5" if conf<0.5 else "0.5-0.7" if conf<0.7 else "0.7-0.8" if conf<0.8 else "0.8-0.9" if conf<0.9 else "0.9-1.0"
                conf_bins[k][0]+=1; conf_bins[k][1]+=int(ex); conf_bins[k][2]+=c
            rec_rows.append({"id":ann.get("id"),"image":img_rel,"split":ann.get("split"),"category":ann.get("category"),"region_idx":gt_idx,"gt_text":gt,"pred_text":pred,"cer":round(c,4),"wer":round(w,4),"exact_match":int(ex),"accent_stripped_cer":round(ac,4),"diacritic_loss_rate":round(loss,4),"confidence":conf,"recognition_ms":rec_ms,"preprocessing":args.preprocessing})
            if c>0 or not matched:
                errors.append({"id":ann.get("id"),"region":gt_idx,"gt":gt,"pred":pred,"cer":round(c,4),"type":"recognition_error" if matched else "detection_miss"})
        # per-image scene row
        # Page text includes ALL predictions (false positives too). GT regions
        # must be annotated in reading order; ignored regions are excluded.
        page_gt="\n".join(gt_texts) if ann.get("regions") else ann.get("text", "")
        page_pred="\n".join(t for t in pred_texts if t)
        if production and pipeline_result is not None and not ann.get('regions'):
            page_pred=pipeline_result.text
        empty_predictions+=int(not page_pred.strip())
        page_c=cer(page_pred,page_gt); page_w=wer(page_pred,page_gt)
        page_cers.append(page_c); page_wers.append(page_w); page_exacts.append(exact_match(page_pred,page_gt))
        scene_rows.append({"id":ann.get("id"),"image":img_rel,"gt_regions":len(gt_polys),"pred_regions":len(pred_polys),"matched":dm["matched"],"missed":dm["missed"],"fp":dm["fp"],"precision":round(dm["precision"],4),"recall":round(dm["recall"],4),"f1":round(dm["f1"],4),"mean_iou":round(dm["mean_iou"],4),"cer":round(page_c,4),"wer":round(page_w,4),"detection_ms":det_ms,"recognition_ms":rec_ms,"total_ms":total_ms})
    n_ann=len(rows_ann)
    n_matched_rec=len(rec_rows)
    avg_cer=round(sum(cer_list)/len(cer_list),4) if cer_list else 0.0
    avg_wer=round(sum(wer_list)/len(wer_list),4) if wer_list else 0.0
    avg_ac=round(sum(accent_list)/len(accent_list),4) if accent_list else 0.0
    avg_loss=round(sum(loss_list)/len(loss_list),4) if loss_list else 0.0
    exact_rate=round(exact_cnt/len(cer_list),4) if cer_list else 0.0
    precision=total_matched/total_pred if total_pred else 0.0
    recall=total_matched/total_gt if total_gt else 0.0
    f1=2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    # confidence bins finalize avg cer etc - simplified
    bins_out={}
    for k,(cnt,exact_count,cer_sum) in conf_bins.items():
        bins_out[k]={"count":cnt,"exact_rate":round(exact_count/cnt,4) if cnt else 0.0,"avg_cer":round(cer_sum/cnt,4) if cnt else 0.0}
    # write CSVs
    with (out_dir/"scene_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["id","image","gt_regions","pred_regions","matched","missed","fp","precision","recall","f1","mean_iou","cer","wer","detection_ms","recognition_ms","total_ms"])
        w.writeheader(); w.writerows(scene_rows)
    with (out_dir/"recognition_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["id","image","split","category","region_idx","gt_text","pred_text","cer","wer","exact_match","accent_stripped_cer","diacritic_loss_rate","confidence","recognition_ms","preprocessing"])
        w.writeheader(); w.writerows(rec_rows)
    with (out_dir/"errors.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["id","region","gt","pred","cer","error","type"])
        w.writeheader()
        for e in errors[:200]:
            w.writerow({k:e.get(k,"") for k in ["id","region","gt","pred","cer","error","type"]})
    # summary
    summary={
        "run_id":run_id,"mode":"scene","recognizer":rec_label,"detector":"paddle_detector","preprocessing":args.preprocessing,
        "data":str(data_dir),"n_images":n_ann,"n_gt_regions":total_gt,"n_pred_regions":total_pred,"n_matched":total_matched,
        "detection":{"precision":round(precision,4),"recall":round(recall,4),"f1":round(f1,4),"matched":total_matched,"missed":total_missed,"fp":total_fp,"iou_threshold":args.iou_threshold},
        "recognition":{"cer":avg_cer,"wer":avg_wer,"exact_match_rate":exact_rate,"accent_stripped_cer":avg_ac,"diacritic_loss_rate":avg_loss,"n_matched":total_matched,"n_scored":n_matched_rec,"includes_detection_misses":True},
        "end_to_end":{"cer":round(sum(page_cers)/len(page_cers),4),"wer":round(sum(page_wers)/len(page_wers),4),"exact_match_rate":round(sum(page_exacts)/len(page_exacts),4),"field_exact_match":round(field_ok/field_total,4) if field_total else None,"aggregation":"macro per page; raw text; all predictions"},
        'field_evaluation': {'evaluated': field_total, 'unmapped': field_unmapped, 'includes_detection_misses': True},
        "psm":args.psm,
        'execution_mode': 'production_pipeline' if production else 'detector_crop_ablation',
        'route': args.route,
        'iou_backend': args.iou_backend,
        'dataset_manifest': manifest,
        'model_artifacts': model_fingerprints([det_model] + [m for lang in languages if (m := resolve_rec_model(lang))], languages),
        'failed_images': len(rows_ann) - successes,
        'empty_predictions': empty_predictions,
        'actual_pipelines': actual_pipelines,
        'runtime_config': {'cpu_threads': settings.cpu_threads, 'detector_side': settings.detector_limit_side_len,
                           'batch_size': settings.tesseract_batch_size, 'document_preprocess': settings.document_preprocess,
                           'hybrid_line_psm': 7},
        'successful_images': successes,
        'timing_scope': 'first image includes lazy initialization; subsequent images warm' if production else 'ablation after explicit initialization',
        "languages":languages,
        "latency_ms":{"model_load_ms":model_load_ms,"avg_detection_ms":round(sum(det_times)/len(det_times),2) if det_times else 0,"avg_recognition_ms":round(sum(rec_times)/len(rec_times),2) if rec_times else 0,"avg_total_ms":round(sum(total_times)/len(total_times),2) if total_times else 0,"cold_detection_ms":first_det_ms,"cold_recognition_ms":first_rec_ms,"warm_avg_detection_ms":round(sum(det_times[1:])/len(det_times[1:]),2) if len(det_times)>1 else 0,"warm_avg_recognition_ms":round(sum(rec_times[1:])/len(rec_times[1:]),2) if len(rec_times)>1 else 0},
        "confidence_bins":bins_out,
        "model_meta":{"python":py_ver,"paddlepaddle":paddle_ver,"paddleocr":po_ver,"detector":det_model,"recognizer":rec_model,"tesseract":tesseract_ver,"device":device,"cli":f"--recognizer {rec_label} --preprocessing {args.preprocessing}","dataset":str(data_dir),"sample_count":n_ann,"git_hash":_git_hash()},
        "selection":{"source":args.source,"split":args.split,"reviewed_only":args.reviewed_only,"include_rejected":args.include_rejected},"dataset_counts":count_records(rows_ann),"warnings":["REAL REVIEWED TEST COUNT <100: SMOKE ONLY NOT REPRESENTATIVE"] if real_reviewed_test_count(rows_ann)<100 else []
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    md=[f"# Scene Benchmark {run_id}\n",f"- recognizer: {rec_label} ({rec_model}) preprocessing={args.preprocessing} iou={args.iou_threshold}\n",f"- data: {data_dir} images={n_ann} gt_regions={total_gt} pred={total_pred} matched={total_matched}\n",f"- detection P={round(precision,3)} R={round(recall,3)} F1={round(f1,3)} | CER={avg_cer} WER={avg_wer} exact={exact_rate} accent_cer={avg_ac} loss={avg_loss}\n",f"- latency load={model_load_ms} cold_det={first_det_ms} cold_rec={first_rec_ms} avg_total={summary['latency_ms']['avg_total_ms']}\n"]
    if summary['warnings']: md.append('\n> WARNING: REAL REVIEWED TEST COUNT <100 — SMOKE ONLY NOT REPRESENTATIVE\n')
    md.append(f"\nPage end-to-end (includes missing and extra text): {summary['end_to_end']}\nRegion recognition includes unmatched GT as empty predictions.\nTesseract PSM: {args.psm}\n")
    md.append(f"\n## Model Meta\n```json\n{json.dumps(summary['model_meta'],ensure_ascii=False,indent=2)}\n```\n")
    md.append(f"\nFormula: diacritic_loss_rate = cer - accent_stripped_cer clamped >=0; IoU threshold={args.iou_threshold} backend={args.iou_backend}; execution={summary['execution_mode']}\n")
    (out_dir/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print(f"Done: {out_dir} P={round(precision,3)} R={round(recall,3)} F1={round(f1,3)} CER={avg_cer} n={n_ann}")
    if summary['warnings']: print(summary['warnings'][0],file=sys.stderr)
    return 0 if successes else 1
if __name__=="__main__": raise SystemExit(main())
