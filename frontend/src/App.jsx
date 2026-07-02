import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Boxes,
  Braces,
  ChevronRight,
  Database,
  FileJson,
  Gauge,
  ImageUp,
  Layers3,
  Loader2,
  Network,
  Play,
  ScanLine,
  UploadCloud,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const navItems = [
  { id: "overview", label: "Overview", icon: Gauge },
  { id: "dataset", label: "Synthetic Dataset", icon: Database },
  { id: "detect", label: "Model Detection", icon: ScanLine },
  { id: "api", label: "API Inference", icon: Network },
];

const workflow = [
  ["Synthetic data", "Generated drawings create image, YOLO label, and semantic JSON pairs."],
  ["YOLO split", "Prepared folders separate train, validation, and test images."],
  ["Color isolation", "HSV filtering removes grayscale floor plans and keeps colored marks."],
  ["Detection and JSON", "FastAPI runs preprocessing, detection, OCR, parsing, and structured export."],
];

function apiUrl(path) {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

function classNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

function App() {
  const [active, setActive] = useState("overview");
  const [dashboard, setDashboard] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [selectedSample, setSelectedSample] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        setLoading(true);
        const [dashRes, sampleRes] = await Promise.all([
          fetch(apiUrl("/api/v1/dashboard")),
          fetch(apiUrl("/api/v1/samples")),
        ]);
        if (!dashRes.ok || !sampleRes.ok) throw new Error("FastAPI did not return dashboard data.");
        const dash = await dashRes.json();
        const samplePayload = await sampleRes.json();
        if (!mounted) return;
        setDashboard(dash);
        setSamples(samplePayload.samples || []);
        setSelectedSampleId((samplePayload.samples || [])[0]?.id || "");
      } catch (err) {
        if (mounted) setError(err.message);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedSampleId) return;
    let mounted = true;
    fetch(apiUrl(`/api/v1/samples/${selectedSampleId}`))
      .then((res) => {
        if (!res.ok) throw new Error("Could not load sample.");
        return res.json();
      })
      .then((payload) => mounted && setSelectedSample(payload))
      .catch((err) => mounted && setError(err.message));
    return () => {
      mounted = false;
    };
  }, [selectedSampleId]);

  const heroImage = selectedSample?.boxed_image_url || samples[0]?.boxed_image_url;

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brandMark"><Boxes size={22} /></span>
          <div>
            <strong>Structural AI</strong>
            <small>Drawing intelligence console</small>
          </div>
        </div>
        <nav className="nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={classNames("navButton", active === item.id && "active")}
                onClick={() => setActive(item.id)}
                title={item.label}
              >
                <Icon size={18} />
                <span>{item.label}</span>
                <ChevronRight size={16} />
              </button>
            );
          })}
        </nav>
        <div className="runtime">
          <span>Runtime</span>
          <code>{API_BASE}</code>
          <span>Weights</span>
          <code>{dashboard?.model?.weight_path || "not trained yet"}</code>
        </div>
      </aside>

      <section className="content">
        <Hero imageUrl={heroImage ? apiUrl(heroImage) : ""} />
        {error && <div className="notice danger">{error}</div>}
        {loading ? (
          <div className="loading"><Loader2 className="spin" /> Warming up the console...</div>
        ) : (
          <>
            {active === "overview" && <Overview dashboard={dashboard} samples={samples} />}
            {active === "dataset" && (
              <Dataset
                samples={samples}
                selectedSample={selectedSample}
                selectedSampleId={selectedSampleId}
                setSelectedSampleId={setSelectedSampleId}
              />
            )}
            {active === "detect" && <Detection />}
            {active === "api" && <ApiInference />}
          </>
        )}
      </section>
    </main>
  );
}

function Hero({ imageUrl }) {
  return (
    <header className="hero">
      <div className="blueprintGrid" />
      <div className="beam beamA" />
      <div className="beam beamB" />
      <div className="heroCopy">
        <h1>Structural Drawing AI Console</h1>
      </div>
      {imageUrl && (
        <div className="heroPreview">
          <img src={imageUrl} alt="Annotated structural drawing preview" />
          <span className="scan" />
        </div>
      )}
    </header>
  );
}

function Overview({ dashboard, samples }) {
  const metrics = dashboard?.metrics || {};
  return (
    <div className="viewStack">
      <section className="metricGrid">
        <Metric icon={ImageUp} label="Sample Images" value={metrics.sample_images ?? 0} />
        <Metric icon={Braces} label="YOLO Labels" value={metrics.yolo_labels ?? 0} />
        <Metric icon={FileJson} label="Semantics" value={metrics.semantics ?? 0} />
        <Metric icon={Activity} label="Trained Weights" value={metrics.trained_weights ? "Ready" : "Missing"} />
      </section>

      <section className="workflow">
        {workflow.map(([title, body], index) => (
          <article className="workflowStep" key={title} style={{ "--i": index }}>
            <span>{index + 1}</span>
            <h3>{title}</h3>
            <p>{body}</p>
          </article>
        ))}
      </section>

      <section className="splitPanel">
        <div>
          <h2>Dataset Split</h2>
          <p>The detector trains on train/images, tunes on val/images, and keeps test/images reserved.</p>
        </div>
        <div className="bars">
          {(dashboard?.splits || []).map((item) => (
            <div className="barRow" key={item.split}>
              <span>{item.split}</span>
              <div><i style={{ width: `${Math.max(4, item.images)}%` }} /></div>
              <strong>{item.images}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="sourceGrid">
        {(dashboard?.sources || []).map((source) => (
          <article className="sourceCard" key={source.source}>
            <Layers3 size={18} />
            <h3>{source.source}</h3>
            <p>{source.images} images, {source.labels} labels, {source.semantics} semantic files</p>
          </article>
        ))}
        <article className="sourceCard accent">
          <Database size={18} />
          <h3>Explorer-ready</h3>
          <p>{samples.length} samples are indexed for React browsing and annotated previews.</p>
        </article>
      </section>
    </div>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <article className="metricCard">
      <Icon size={20} />
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function Dataset({ samples, selectedSample, selectedSampleId, setSelectedSampleId }) {
  const visibleSamples = samples.slice(0, 18);
  return (
    <div className="datasetLayout">
      <section className="sampleRail">
        <h2>Sample Explorer</h2>
        <select value={selectedSampleId} onChange={(event) => setSelectedSampleId(event.target.value)}>
          {samples.map((sample) => (
            <option value={sample.id} key={sample.id}>{sample.source} / {sample.filename}</option>
          ))}
        </select>
        <div className="thumbGrid">
          {visibleSamples.map((sample) => (
            <button
              key={sample.id}
              className={classNames("thumb", selectedSampleId === sample.id && "selected")}
              onClick={() => setSelectedSampleId(sample.id)}
              title={`${sample.source} ${sample.filename}`}
            >
              <img src={apiUrl(sample.boxed_image_url)} alt="" />
            </button>
          ))}
        </div>
      </section>

      <section className="sampleStage">
        {selectedSample ? (
          <>
            <div className="stageImage">
              <img src={apiUrl(selectedSample.boxed_image_url)} alt={selectedSample.filename} />
            </div>
            <div className="sampleMeta">
              <div>
                <span className="eyebrow">Rendered sample</span>
                <h2>{selectedSample.filename}</h2>
                <p>{selectedSample.source}</p>
              </div>
              <JsonBlock title="Semantics" value={selectedSample.semantics} />
            </div>
            <DataTable rows={selectedSample.labels || []} />
          </>
        ) : (
          <div className="notice">No samples found. Run the synthetic data generator first.</div>
        )}
      </section>
    </div>
  );
}

function DataTable({ rows }) {
  if (!rows.length) return <div className="notice">No labels found for this sample.</div>;
  const hasSource = rows.some((row) => row.source);
  const hasText = rows.some((row) => row.text);
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Class</th>
            {hasSource && <th>Source</th>}
            {hasText && <th>Text</th>}
            <th>X Center</th>
            <th>Y Center</th>
            <th>Width</th>
            <th>Height</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.class_name}-${index}`}>
              <td>{row.class_name}</td>
              {hasSource && <td>{row.source || "-"}</td>}
              {hasText && <td>{row.text || "-"}</td>}
              <td>{row.x_center.toFixed(3)}</td>
              <td>{row.y_center.toFixed(3)}</td>
              <td>{row.width.toFixed(3)}</td>
              <td>{row.height.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Detection() {
  return <DetectionWorkflow />;
}

function DetectionWorkflow() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const fileUrl = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => () => fileUrl && URL.revokeObjectURL(fileUrl), [fileUrl]);

  function onFileChange(event) {
    const nextFile = event.target.files?.[0];
    setFile(nextFile || null);
    setResult(null);
    setMessage("");
    setPreview(nextFile?.type?.startsWith("image/") ? URL.createObjectURL(nextFile) : null);
  }

  async function runDetection() {
    if (!file) return;
    setBusy(true);
    setMessage("");
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(apiUrl("/api/v1/detect/image"), { method: "POST", body: form });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
      setResult(payload);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setBusy(false);
    }
  }

  const formattedRows = result?.detections?.map((det) => {
    const [x1, y1, x2, y2] = det.bbox;
    return {
      class_name: `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`,
      source: det.source || "yolo",
      text: det.text || "",
      x_center: (x1 + x2) / 2,
      y_center: (y1 + y2) / 2,
      width: x2 - x1,
      height: y2 - y1,
    };
  }) || [];

  return (
    <section className="uploadLayout">
      <div className="uploadCard">
        <span className="eyebrow"><ScanLine size={15} /> Detection workflow</span>
        <h2>Model Detection</h2>
        <p>Upload a drawing image to preview the color-isolation preprocessing and run the fast YOLO detector to draw bounding boxes and masks.</p>
        <label className="dropZone">
          <input type="file" accept="image/png,image/jpeg" onChange={onFileChange} />
          <UploadCloud size={34} />
          <strong>{file ? file.name : "Choose a drawing file"}</strong>
          <span>image/png / image/jpeg</span>
        </label>
        <button className="primaryButton" onClick={runDetection} disabled={!file || busy}>
          {busy ? <Loader2 className="spin" size={18} /> : <ScanLine size={18} />}
          Run detection
        </button>
        {message && <div className="notice danger">{message}</div>}
      </div>

      <div className="previewStack">
        {preview && !result && <img className="uploadedPreview" src={preview} alt="Uploaded drawing preview" />}
        {result && (
          <>
            <div className="preprocessGrid">
              <PreviewTile title="Original" src={result.original} />
              <PreviewTile title="HSV mask" src={result.mask} />
              <PreviewTile title="Cleaned" src={result.cleaned} />
              <div className="retained">
                <strong>{result.colored_pixel_count.toLocaleString()}</strong>
                <span>colored pixels retained</span>
                <em>{(result.retained_ratio * 100).toFixed(2)}%</em>
              </div>
            </div>
            <div className="stageImage">
              <img src={result.rendered} alt="YOLO detections" />
            </div>
            {result.summary && (
              <div className="detectionSummary">
                <span><strong>{result.summary.beams}</strong> beams</span>
                <span><strong>{result.summary.text}</strong> text boxes</span>
                <span><strong>{result.summary.ocr_text}</strong> OCR-added</span>
              </div>
            )}
            <DataTable rows={formattedRows} />
          </>
        )}
      </div>
    </section>
  );
}

function ApiInference() {
  const [mode, setMode] = useState("image");
  return (
    <section className="apiPanel">
      <div className="segmented">
        <button className={mode === "image" ? "on" : ""} onClick={() => setMode("image")}>Image</button>
        <button className={mode === "pdf" ? "on" : ""} onClick={() => setMode("pdf")}>PDF</button>
      </div>
      <UploadWorkflow
        key={mode}
        title={mode === "image" ? "Image Inference" : "PDF Batch Inference"}
        body={mode === "image" ? "Send a PNG or JPG to FastAPI and inspect the structured JSON response." : "Send a structural PDF to FastAPI. The backend renders every page and processes each one."}
        accept={mode === "image" ? "image/png,image/jpeg" : "application/pdf"}
        endpoint={mode === "image" ? "/api/v1/inference/image" : "/api/v1/inference/pdf"}
        preprocess={mode === "image"}
      />
    </section>
  );
}

function UploadWorkflow({ title, body, accept, endpoint, preprocess }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [preprocessData, setPreprocessData] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const fileUrl = useMemo(() => {
    if (!file || file.type === "application/pdf") return "";
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => () => fileUrl && URL.revokeObjectURL(fileUrl), [fileUrl]);

  async function onFileChange(event) {
    const nextFile = event.target.files?.[0];
    setFile(nextFile || null);
    setResult(null);
    setPreprocessData(null);
    setMessage("");
    setPreview(nextFile?.type?.startsWith("image/") ? URL.createObjectURL(nextFile) : null);
    if (nextFile && preprocess) {
      const form = new FormData();
      form.append("file", nextFile);
      try {
        const res = await fetch(apiUrl("/api/v1/preprocess/image"), { method: "POST", body: form });
        if (!res.ok) throw new Error("Preprocessing failed.");
        setPreprocessData(await res.json());
      } catch (err) {
        setMessage(err.message);
      }
    }
  }

  async function runInference() {
    if (!file) return;
    setBusy(true);
    setMessage("");
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(apiUrl(endpoint), { method: "POST", body: form });
      const text = await res.text();
      let payload;
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { raw: text };
      }
      if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
      setResult(payload);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="uploadLayout">
      <div className="uploadCard">
        <span className="eyebrow"><UploadCloud size={15} /> Upload workflow</span>
        <h2>{title}</h2>
        <p>{body}</p>
        <label className="dropZone">
          <input type="file" accept={accept} onChange={onFileChange} />
          <UploadCloud size={34} />
          <strong>{file ? file.name : "Choose a drawing file"}</strong>
          <span>{accept.replaceAll(",", " / ")}</span>
        </label>
        <button className="primaryButton" onClick={runInference} disabled={!file || busy}>
          {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Run inference
        </button>
        {message && <div className="notice danger">{message}</div>}
      </div>

      <div className="previewStack">
        {preview && <img className="uploadedPreview" src={preview} alt="Uploaded drawing preview" />}
        {preprocessData && (
          <div className="preprocessGrid">
            <PreviewTile title="Original" src={preprocessData.original} />
            <PreviewTile title="HSV mask" src={preprocessData.mask} />
            <PreviewTile title="Cleaned" src={preprocessData.cleaned} />
            <div className="retained">
              <strong>{preprocessData.colored_pixel_count.toLocaleString()}</strong>
              <span>colored pixels retained</span>
              <em>{(preprocessData.retained_ratio * 100).toFixed(2)}%</em>
            </div>
          </div>
        )}
        {result && <InferenceResult value={result} />}
      </div>
    </section>
  );
}

function PreviewTile({ title, src }) {
  return (
    <article className="previewTile">
      <img src={src} alt={title} />
      <span>{title}</span>
    </article>
  );
}

function JsonBlock({ title, value }) {
  return (
    <section className="jsonBlock">
      <h3>{title}</h3>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

function InferenceResult({ value }) {
  const data = value?.data || value;
  const elements = data?.elements || [];
  const summary = data?.summary || {};

  return (
    <section className="inferenceResult">
      <div className="resultHeader">
        <div>
          <h3>Inference Response</h3>
          <span>{data?.drawing_id || "uploaded drawing"}</span>
        </div>
        <div className="resultStats">
          <strong>{summary.element_count ?? elements.length}</strong>
          <span>elements</span>
          <strong>{summary.annotation_count ?? elements.reduce((total, el) => total + (el.annotations?.length || 0), 0)}</strong>
          <span>texts</span>
        </div>
      </div>

      <div className="elementList">
        {elements.length === 0 && <div className="notice">No structural elements were returned.</div>}
        {elements.map((element, index) => (
          <article className="elementRow" key={element.id || index}>
            <div className="elementMeta">
              <strong>{element.id || `${element.type}_${index + 1}`}</strong>
              <span>{element.type}</span>
              <em>{Math.round((element.detection_confidence || 0) * 100)}%</em>
            </div>
            <div className="annotationList">
              {(element.annotations || []).length === 0 && <span className="emptyAnnotation">No associated text</span>}
              {(element.annotations || []).map((ann, annIndex) => (
                <div className="annotationPill" key={`${ann.text}-${annIndex}`}>
                  <strong>{ann.normalized_text || ann.text}</strong>
                  <span>
                    Dia {ann.parsed?.diameter ?? "-"} · Spacing {ann.parsed?.spacing ?? "-"} · Layer {ann.parsed?.layer ?? "-"}
                  </span>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>

      <JsonBlock title="Raw JSON" value={value} />
    </section>
  );
}

export default App;
