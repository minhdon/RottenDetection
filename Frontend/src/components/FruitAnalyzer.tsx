import React, { ChangeEvent, DragEvent, useRef, useState } from "react";

const API_URL = "http://localhost:8000/predict/upload"; // Đã đổi port thành 8000 theo đúng Backend của bạn

const MODELS = [
  { value: "rf_rgb", label: "Random Forest — RGB" },
  { value: "xgb_rgb", label: "XGBoost — RGB" },
  { value: "resnet_xgb", label: "ResNet + XGBoost (Deep Learning)" },
  // 👇 Đã thêm tùy chọn ResNet + Random Forest vào đây
  { value: "resnet_rf", label: "ResNet + Random Forest" }, 
];

// --- ĐỊNH NGHĨA MAPPING TỪ SỐ SANG CHỮ ---
const FRUIT_MAP: Record<string, string> = {
  "0": "Táo (Apple)",
  "1": "Chuối (Banana)",
  "2": "Cam (Orange)",
  "3": "Nho (Grape)",
};

const QUALITY_MAP: Record<string, string> = {
  "0": "Sạch / Tươi (Fresh)",
  "1": "Bị hư / Thối (Rotten)",
};

// --- ĐỊNH NGHĨA KIỂU DỮ LIỆU TRẢ VỀ TỪ API ---
interface PredictionResponse {
  filename: string;
  pipeline_used: string;
  predictions: {
    fruit_type: string | number;
    quality: string | number;
  };
}

// Hàm xác định màu sắc dựa trên kết quả chất lượng
const qualityTone = (q: string | number) => {
  const status = String(q);
  if (status === "0") return "bg-green-100 text-green-800 border-green-300"; // Tươi -> Xanh
  if (status === "1") return "bg-red-100 text-red-800 border-red-300";       // Hư -> Đỏ
  return "bg-gray-100 text-gray-800 border-gray-300";
};

export const FruitAnalyzer = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [modelType, setModelType] = useState<string>("resnet_xgb");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File | null | undefined) => {
    if (!f) return;
    if (!["image/jpeg", "image/jpg", "image/png"].includes(f.type)) {
      setError("Vui lòng tải lên ảnh định dạng .jpg, .jpeg hoặc .png");
      return;
    }
    setError(null);
    setResult(null);
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => handleFile(e.target.files?.[0]);
  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("model_type", modelType);
      
      const res = await fetch(API_URL, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Lỗi Server: ${res.status} ${res.statusText}`);
      
      const data = (await res.json()) as PredictionResponse;
      setResult(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Lỗi không xác định";
      setError(`Không thể kết nối đến Backend. ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-slate-50 px-4 py-12 md:py-20 font-sans text-slate-900">
      <div className="mx-auto max-w-5xl">
        {/* Header */}
        <header className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-4 py-1.5 text-xs font-medium text-blue-700 mb-6 shadow-sm">
            <span>✨ Powered by AI Machine Learning</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900">
            Hệ thống Nhận diện <br />
            <span className="text-blue-600">Chất lượng Trái cây</span>
          </h1>
          <p className="mt-4 text-base md:text-lg text-slate-600 max-w-xl mx-auto">
            Tải ảnh lên và để các mô hình AI (Random Forest, XGBoost, ResNet) dự đoán loại quả và độ tươi ngon.
          </p>
        </header>

        <div className="grid gap-6 md:grid-cols-5">
          {/* Cột trái: Upload & Controls */}
          <section className="md:col-span-3 rounded-2xl bg-white shadow-lg border border-slate-200 p-6 md:p-8">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-800">
              🖼️ Tải ảnh lên
            </h2>

            {/* Vùng kéo thả ảnh */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={`relative cursor-pointer rounded-xl border-2 border-dashed transition-all overflow-hidden min-h-[260px] flex items-center justify-center text-center p-6 ${
                dragOver ? "border-blue-500 bg-blue-50" : "border-slate-300 hover:border-blue-400 hover:bg-slate-50"
              }`}
            >
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" className="hidden" onChange={onInputChange} />
              
              {preview ? (
                <>
                  <img src={preview} alt="Preview" className="max-h-80 w-auto rounded-lg object-contain shadow-md" />
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); clearFile(); }}
                    className="absolute top-3 right-3 rounded-full bg-red-100 p-2 text-red-600 hover:bg-red-500 hover:text-white shadow-sm transition-colors"
                    title="Xóa ảnh"
                  >
                    ✕
                  </button>
                </>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div className="rounded-full bg-blue-100 p-4 text-blue-600 text-3xl">
                    📁
                  </div>
                  <div>
                    <p className="font-semibold text-slate-700">Kéo & thả ảnh vào đây</p>
                    <p className="text-sm text-slate-500 mt-1">hoặc click để chọn tệp (JPG, PNG)</p>
                  </div>
                </div>
              )}
            </div>

            {/* Chọn mô hình AI */}
            <div className="mt-6 space-y-2">
              <label htmlFor="model" className="text-sm font-semibold text-slate-700">Chọn Pipeline Mô hình</label>
              <select 
                id="model" 
                value={modelType} 
                onChange={(e) => setModelType(e.target.value)}
                className="w-full h-11 px-3 py-2 bg-white border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {MODELS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>

            {/* Nút Submit */}
            <button
              onClick={submit}
              disabled={!file || loading}
              className={`mt-6 w-full h-12 text-base font-bold rounded-xl shadow-md transition-all flex items-center justify-center gap-2 ${
                !file || loading 
                  ? "bg-slate-300 text-slate-500 cursor-not-allowed" 
                  : "bg-blue-600 text-white hover:bg-blue-700 hover:shadow-lg"
              }`}
            >
              {loading ? "⏳ Đang phân tích..." : "🚀 Phân tích ngay"}
            </button>

            {/* Thông báo Lỗi */}
            {error && (
              <div className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex items-start gap-2">
                <span>⚠️</span>
                <p>{error}</p>
              </div>
            )}
          </section>

          {/* Cột phải: Hiển thị Kết quả */}
          <section className="md:col-span-2 rounded-2xl bg-white shadow-lg border border-slate-200 p-6 md:p-8">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-800">
              📊 Kết quả dự đoán
            </h2>

            {!result && !loading && (
              <div className="flex flex-col items-center justify-center text-center h-full min-h-[260px] text-slate-400 bg-slate-50 rounded-xl border border-dashed border-slate-200">
                <span className="text-4xl mb-3">🤖</span>
                <p className="text-sm">Kết quả phân tích sẽ hiển thị ở đây.</p>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center h-full min-h-[260px] text-blue-600">
                <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-3"></div>
                <p className="text-sm font-medium animate-pulse">Đang chạy mô hình...</p>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                {/* Loại quả */}
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <p className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">Loại trái cây</p>
                  <p className="text-2xl font-black text-blue-700">
                    {FRUIT_MAP[String(result.predictions.fruit_type)] || `Loại số: ${result.predictions.fruit_type}`}
                  </p>
                </div>

                {/* Chất lượng */}
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <p className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Tình trạng (Chất lượng)</p>
                  <span className={`px-4 py-1.5 rounded-full text-sm font-bold border ${qualityTone(result.predictions.quality)}`}>
                    {QUALITY_MAP[String(result.predictions.quality)] || `Trạng thái: ${result.predictions.quality}`}
                  </span>
                </div>

                {/* Chi tiết file */}
                <div className="pt-4 border-t border-slate-200 space-y-3 text-sm">
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-slate-500 font-medium">Tên file ảnh:</span>
                    <span className="font-semibold text-slate-800 truncate max-w-[60%]">{result.filename}</span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-slate-500 font-medium">Pipeline đã dùng:</span>
                    <span className="font-mono text-xs font-bold bg-slate-200 text-slate-700 px-2 py-1 rounded">
                      {result.pipeline_used}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};