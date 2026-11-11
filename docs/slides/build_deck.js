// สร้างสไลด์ฉบับสุดท้าย CN340 จาก docs/slides/.build/slides_data.json (สร้างโดย experiments/60_make_slides.py)
//   NODE_PATH=docs/slides/.build/node_modules node docs/slides/build_deck.js
// → docs/slides/CN340_final_presentation.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, ".build", "slides_data.json"), "utf8"));
const OUT = path.join(__dirname, "CN340_final_presentation.pptx");
const SKILL_THEME = process.env.APPLY_THEME_JS;

const COL = { bg: "FDF0E5", main: "9B1039", teal: "0F7681", text: "2F2A28", dark: "262A33", muted: "6B625C", line: "E3CDBB", card: "FFFFFF", soft: "F6E2D3", tealSoft: "D7ECEC", good: "0F7681", bad: "9B1039" };
const THEME = { name: "CN340 Final", colors: { dk1: COL.text, lt1: "FFFFFF", dk2: COL.dark, lt2: COL.bg, accent1: COL.main, accent2: COL.teal, accent3: "C8955F", accent4: "5E6B7A", accent5: "D98A9E", accent6: "7FB5B5", hlink: COL.teal, folHlink: COL.main } };
const F = "Prompt";
const NL = String.fromCharCode(10);
const W = 13.333, H = 7.5, MX = 0.6;

// ---------- รูปแบบตัวเลข ----------
const sgn = (v, d = 3) => (v === null || v === undefined ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d));
const num = (v, d = 3) => (v === null || v === undefined ? "–" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(d));
const pct = (v, d = 1) => (v === null || v === undefined ? "ไม่นิยาม" : (v * 100).toFixed(d) + "%");
const ppt = (v, d = 1) => (v === null || v === undefined ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toFixed(d));
const ci = (lo, hi, d = 3) => `[${sgn(lo, d)}, ${sgn(hi, d)}]`;
const thDate = (s) => { const [y, m, d] = s.split("-").map(Number); const mo = ["", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."][m]; return `${d} ${mo} ${y + 543}`; };
const comma = (v) => Math.round(v).toLocaleString("en-US");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: F, bodyFontFace: F };
pres.title = "การพยากรณ์ทิศทางและราคาหุ้น KBANK และ ADVANC";
pres.author = "ทีม CN340";

pres.defineSlideMaster({
  title: "TITLE", background: { color: COL.dark },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: MX, y: 2.2, w: 12.1, h: 1.6, fontFace: F, fontSize: 38, bold: true, color: "FFFFFF", valign: "bottom", margin: 0 }, text: "" } },
            { placeholder: { options: { name: "body", type: "body", x: MX, y: 4.0, w: 12.1, h: 1.4, fontFace: F, fontSize: 18, color: "F2D9C4", valign: "top", margin: 0 }, text: "" } },
            { text: { text: "ที่มา: github.com/hansera1/CN340-Final-StockPrediction · results/ และ outputs/ ในเครื่อง", options: { x: MX, y: 7.0, w: 11.2, h: 0.32, fontFace: F, fontSize: 9, color: "B9AFA8", margin: 0 } } }],
  slideNumber: { x: 12.15, y: 7.0, w: 0.6, h: 0.32, fontFace: F, fontSize: 10, color: "B9AFA8", align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: COL.bg },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: MX, y: 0.3, w: 12.1, h: 0.75, fontFace: F, fontSize: 26, bold: true, color: COL.main, valign: "middle", margin: 0 }, text: "" } },
            { placeholder: { options: { name: "src", type: "body", x: MX, y: 7.0, w: 11.2, h: 0.32, fontFace: F, fontSize: 9, color: COL.muted, valign: "middle", margin: 0 }, text: "" } }],
  slideNumber: { x: 12.15, y: 7.0, w: 0.6, h: 0.32, fontFace: F, fontSize: 10, color: COL.muted, align: "right" },
});

let page = 0;
const SECTIONS = new Set();
function sec(t) { if (!SECTIONS.has(t)) { pres.addSection({ title: t }); SECTIONS.add(t); } return t; }
let SIG_PAGE = 0;
function slide(title, src, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec(section) });
  page += 1;
  s.addText(title, { placeholder: "title" });
  s.addText("ที่มา: " + src, { placeholder: "src" });
  return s;
}
const txt = (s, t, o) => s.addText(t, { fontFace: F, color: COL.text, fontSize: 14, isTextBox: true, valign: "top", margin: 0.05, ...o });
function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })),
    { fontFace: F, color: COL.text, fontSize: 14, isTextBox: true, valign: "top", margin: 0.05, ...o });
}
function card(s, x, y, w, h, head, body, color = COL.main, bodySize = 13) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: COL.card }, line: { color: COL.line, width: 0.75 }, rectRadius: 0.08, objectName: "card" });
  s.addText(head, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: 0.42, fontFace: F, fontSize: 15, bold: true, color, isTextBox: true, margin: 0, valign: "top" });
  s.addText(body, { x: x + 0.18, y: y + 0.58, w: w - 0.36, h: h - 0.7, fontFace: F, fontSize: bodySize, color: COL.text, isTextBox: true, margin: 0, valign: "top" });
}
function stat(s, x, y, w, big, label, color = COL.main) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.6, fill: { color: COL.card }, line: { color: COL.line, width: 0.75 }, rectRadius: 0.08 });
  s.addText(big, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.75, fontFace: F, fontSize: 30, bold: true, color, isTextBox: true, margin: 0, align: "left" });
  s.addText(label, { x: x + 0.15, y: y + 0.88, w: w - 0.3, h: 0.65, fontFace: F, fontSize: 12, color: COL.text, isTextBox: true, margin: 0 });
}
function table(s, rows, o = {}) {
  const fs_ = o.fontSize || 11;
  const head = rows[0].map((c) => ({ text: c, options: { bold: true, color: "FFFFFF", fill: { color: o.headColor || COL.main }, fontSize: fs_, fontFace: F, valign: "middle" } }));
  const body = rows.slice(1).map((r, i) => r.map((c) => {
    const cell = typeof c === "object" && c !== null && c.text !== undefined ? c : { text: String(c) };
    return { text: cell.text, options: { fontSize: fs_, fontFace: F, color: COL.text, valign: "middle", fill: { color: i % 2 ? "FBF6F1" : "FFFFFF" }, ...(cell.options || {}) } };
  }));
  s.addTable([head, ...body], { x: o.x ?? MX, y: o.y ?? 1.25, w: o.w ?? 12.1, colW: o.colW, rowH: o.rowH ?? 0.32, border: { type: "solid", pt: 0.5, color: COL.line }, margin: 0.05, autoPage: false });
}
const sigNote = (s, y = 6.55, extra = "") => txt(s, `ตัวเลขในวงเล็บคือช่วงความเชื่อมั่น (CI) ตามระดับที่ระบุ จาก block bootstrap รายวัน 2,000 ครั้ง · ถือว่ามีนัยสำคัญเมื่อ CI ไม่ครอบศูนย์ · วิธีละเอียดในหน้า ${SIG_PAGE}${extra}`, { x: MX, y, w: 12.1, h: 0.4, fontSize: 10, color: COL.muted, italic: true });
const chartText = { catAxisLabelFontFace: F, valAxisLabelFontFace: F, dataLabelFontFace: F, legendFontFace: F, titleFontFace: F, catAxisLabelColor: COL.muted, valAxisLabelColor: COL.muted, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, legendFontSize: 10, legendColor: COL.text, titleColor: COL.text, titleFontSize: 12 };

// ======================================================================
// 1. หน้าปก
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: sec("บทนำ") });
  page += 1;
  s.addText("การพยากรณ์ทิศทางและราคาหุ้น KBANK และ ADVANC", { placeholder: "title" });
  s.addText([{ text: "รายวิชา CN340 · การนำเสนอฉบับสุดท้าย", options: { breakLine: true } },
             { text: `ผลการส่งจริงอัปเดตถึงวันที่ ${thDate(D.live.last_day)} (${D.live.days} จาก 5 วันทำการ)` }], { placeholder: "body" });
}

// 2. ที่มาและโจทย์
{
  const s = slide("ที่มาและโจทย์ของงาน", "configs/rules.yaml · docs/01_โจทย์และกติกา.md", "ที่มาและปัญหา");
  txt(s, "รายวิชา CN340 กำหนดให้พยากรณ์หุ้น KBANK (ธนาคารกสิกรไทย) และ ADVANC (แอดวานซ์ อินโฟร์ เซอร์วิส) ในตลาดหลักทรัพย์แห่งประเทศไทย วันละ 2 รอบ ด้วยโมเดลการเรียนรู้ของเครื่อง (machine learning) และการเรียนรู้เชิงลึก (deep learning)", { x: MX, y: 1.2, w: 12.1, h: 0.8, fontSize: 15 });
  table(s, [["รอบ", "ส่งก่อน", "จุดตัดข้อมูล", "ทิศทาง (ขึ้น/ไม่ขึ้น)", "ราคาที่ต้องทาย"],
            ["เช้า", "09:00", "08:30", "P(12:30) เทียบ P(10:00)", "P(12:30)"],
            ["บ่าย", "13:00", "12:35", "ราคาปิดทางการ (รวม ATC) เทียบ P(14:00)", "ราคาปิดทางการ (รวม ATC)"]], { y: 2.15, colW: [1.2, 1.4, 1.8, 4.3, 3.4], rowH: 0.45, fontSize: 13 });
  card(s, MX, 3.85, 3.9, 2.4, "สิ่งที่ต้องส่งแต่ละรอบ", "ทิศทาง (ขึ้นหรือไม่ขึ้น) ราคาคาด และช่วงราคา 50% และ 80% · วันที่ราคาปลายเท่ากับราคาต้นรอบนับเป็น \"ไม่ขึ้น\"");
  card(s, 4.7, 3.85, 3.9, 2.4, "ช่วงส่งจริง", "9–13 พฤศจิกายน 2569 รวม 20 คำทายทิศทาง (5 วัน × 2 หุ้น × 2 รอบ)", COL.teal);
  card(s, 8.8, 3.85, 3.9, 2.4, "สัญลักษณ์ที่ใช้", "P(t) คือราคาซื้อขายล่าสุด ณ เวลา t · P(c) คือราคาล่าสุดที่รู้ ณ จุดตัดข้อมูล (รอบเช้าคือราคาปิดทางการวันก่อน)");
}

// 3. ลักษณะที่ทำให้ยาก
{
  const s = slide("ลักษณะของปัญหาที่ทำให้พยากรณ์ได้ยาก", "Fama (1970) · results/e14/kbank_flat_rate.csv · ชุดข้อมูล _wf", "ที่มาและปัญหา");
  const fl = D.data.flat_range;
  const items = [
    ["ราคาใกล้การเดินสุ่ม (random walk)", "ตามทฤษฎีตลาดมีประสิทธิภาพ (Fama, 1970) สัญญาณที่ทายได้จึงอ่อน → รับมือด้วยตัวเทียบทางสถิติและการทดสอบนัยสำคัญที่เข้มงวด"],
    ["ราคาขยับเป็นช่องราคา (tick)", `วันที่ราคาเท่าเดิมมี ${pct(fl[0], 0)}–${pct(fl[1], 0)} ของรอบ แยกตามหุ้นและรอบ → ใช้โมเดลวันนิ่งและเป้าหมายขึ้น/ไม่ขึ้น`],
    ["ข้อมูลระหว่างวันย้อนหลังจำกัด", `ใช้แท่ง 60 นาทีเป็นข้อมูลหลัก ${comma(D.data.n_days)} วันทำการ (${thDate(D.data.first_day)} – ${thDate(D.data.last_day)})`],
    ["ต้องส่งก่อนตลาดเปิดและระหว่างพัก", "กำหนดจุดตัดข้อมูลให้ใช้เฉพาะข้อมูลที่รู้ก่อนส่ง และสร้างระบบส่งอัตโนมัติ"]];
  items.forEach(([h, b], i) => card(s, MX + (i % 2) * 6.15, 1.3 + Math.floor(i / 2) * 2.7, 5.95, 2.45, h, b, i % 2 ? COL.teal : COL.main, 14));
}

// 4. ปัญหาที่พบระหว่างพัฒนา
const PROBS = [
  ["A1", "เป้าหมายที่ใช้ฝึกไม่ตรงวิธีให้คะแนน (ขึ้น/ลง ตัดวันนิ่ง เทียบกติกาขึ้น/ไม่ขึ้น)", "โมเดลที่ดีบนเป้าหมายหนึ่งเสียเปรียบบนอีกเป้าหมาย", "เพิ่ม acc_up_notup และทดลองเป้าหมายขึ้น/ไม่ขึ้น (E16) แล้วเลือกแยกรอบ", "E16 ไม่ผ่าน จึงคง A-ud รอบเช้า และใช้ E15 MLP รอบบ่าย"],
  ["A2", "ราคาเปิด 10:00 จาก yfinance ไม่ใช่ราคาประมูลจริง", "ป้ายคำตอบอาจผิด", "ใช้ราคาเปิดรายวันจาก Settrade", "ราคา 4 จุดของโจทย์ตรงกับ Settrade 1 นาที 32/32 จุด"],
  ["A3", "ภาคบ่ายเลื่อนเปิดจาก 14:30 เป็น 14:00 (25 มี.ค. 2567)", "ข้อมูลรอบบ่ายก่อนวันนั้นใช้ไม่ได้", "ใช้รอบบ่ายตั้งแต่ 25 มี.ค. 2567", "ป้ายรอบบ่ายสอดคล้องกับเวลาจริง"],
  ["A4", "Settrade ให้แท่ง 1 นาทีย้อนหลังได้ราว 4 วันทำการ", "ทดสอบโมเดลข้อมูลละเอียดได้ไม่พอ", "ใช้แท่ง 60 นาทีเป็นหลัก และเก็บแท่ง 1 นาทีสะสมทุกวัน", "E19 / E23b ยังเป็นงานสำรวจ (ข้อมูล 10 วัน)"],
  ["A5", "CNN overfit ตั้งแต่ epoch ต้น ๆ", "ไม่มี CNN ใดชนะตัวเทียบ", "ตรึง backbone, regularization แรงขึ้น, ยืมหุ้น SET50 อีก 20 ตัว", "overfit ลดลง แต่ฝีมือไม่เพิ่ม (E20)"],
  ["A6", "KBANK เปลี่ยนช่องราคาจาก 0.50 เป็น 1.00 บาท", "สัดส่วนวันนิ่งเปลี่ยน ข้อมูลช่วงใหม่มีน้อย", "นิยามช่องราคาต่อวันด้วย tick_size(P(c)) และทดลอง C′/C-adv", "ไม่ผ่านเกณฑ์ (CI กว้าง)"],
  ["A7", "นิยาม P(c) และจุดตัดข้อมูลไม่ตรงสถานการณ์จริง", "ต้องแก้นิยามและรันใหม่", "P(c) รอบเช้า = ราคาปิดทางการ · จุดตัด 08:30 / 12:35", "backtest ตรงกับวันจริง"],
  ["A8", "รอบบ่ายเปลี่ยนเป็นราคาปิดทางการรวม ATC และเส้นตายเปลี่ยน", "ต้องสร้างป้ายและฝึกรอบบ่ายใหม่", "สร้างป้าย ATC และล็อกโมเดลชุดใหม่", "ใช้ส่งจริงตั้งแต่ 9 พ.ย. 2569"],
  ["A9", "ทดลองหลายสิบแบบ เสี่ยงเลือกแบบที่ดีโดยบังเอิญ", "ผลดูดีเกินจริง", "ลงทะเบียนเกณฑ์ล่วงหน้า, Bonferroni, PBO, ตัวแทนจาก inner CV", `นับการเทียบสะสมได้ ${D.ledger_total} ครั้ง`],
  ["A10", "Settrade อนุญาต login ทีละ session", "สองเครื่องอาจชนกัน", "VM หลัก + โน้ตบุ๊กสำรอง ห้าม login จากโน้ตบุ๊ก 08:25–08:55 และ 12:30–12:45", "ส่งจริงไม่ชนกัน"],
  ["A11", "backtest ราคาไม่หักเงินปันผลวัน XD ขณะที่ระบบจริงหัก", "RMSE ใน backtest สูงกว่าระบบจริง", "ระบุเป็นข้อจำกัด และวัดผล (E27)", "RMSE รอบเช้าลดจาก 3.083 เหลือราว 2.978 เมื่อหัก"],
  ["A12", "โมเดลเป้าหมายขึ้น/ไม่ขึ้นทาย \"ไม่ขึ้น\" เกือบทุกรอบ", "ความแม่นยำสูงมาจากอัตราพื้นฐาน", "รายงาน sensitivity, balanced accuracy และเทียบ E0-un", "เห็นชัดว่าไม่ได้แยกวันขึ้น"]];
{
  const s = slide("ปัญหาที่พบระหว่างพัฒนาโมเดล", "รายงานรอบ 2–7 (docs/reports/) · results/e14–e28 · docs/02_ข้อมูลและfeatures.md", "ที่มาและปัญหา");
  table(s, [["#", "ปัญหา", "ผลกระทบ"], ...PROBS.map((p) => [p[0], p[1], p[2]])], { y: 1.15, colW: [0.6, 7.0, 4.5], rowH: 0.41, fontSize: 11 });
  txt(s, `วิธีแก้และผลหลังแก้อยู่ในหน้า ${page + 17}`, { x: MX, y: 6.62, w: 6, h: 0.3, fontSize: 10, color: COL.muted, italic: true });
}

// 5. วัตถุประสงค์
{
  const s = slide("วัตถุประสงค์", "docs/prompt_slides_final.md ข้อ 2", "ที่มาและปัญหา");
  const obj = ["พัฒนาโมเดลพยากรณ์ทิศทาง (ขึ้น/ไม่ขึ้น) ของ KBANK และ ADVANC ทั้งสองรอบ ที่แม่นยำเหนือตัวเทียบทางสถิติอย่างมีนัยสำคัญ ภายใต้การทดสอบแบบเดินหน้า (walk-forward validation)",
    "พัฒนาโมเดลพยากรณ์ราคาและช่วงราคา ที่มีค่ารากที่สองของความคลาดเคลื่อนกำลังสองเฉลี่ย (RMSE) ไม่สูงกว่าการใช้ราคาล่าสุด และช่วงราคาครอบคลุมราคาจริงตามระดับที่กำหนด",
    "ประเมินว่าข้อมูลความละเอียดสูง (5 นาที/1 นาที) และภาพ scalogram กับ CNN ที่ฝึกล่วงหน้า (pretrained) ช่วยการพยากรณ์ได้หรือไม่",
    "สร้างระบบส่งผลอัตโนมัติที่ทำงานได้จริงบนเครื่องคลาวด์ตลอดสัปดาห์ส่งจริง",
    "ประเมินมูลค่าทางการเงินของสัญญาณทิศทางด้วย XIRR และ Sharpe ratio"];
  obj.forEach((t, i) => {
    const y = 1.2 + i * 1.08;
    s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.08, w: 0.6, h: 0.6, fill: { color: i % 2 ? COL.teal : COL.main }, line: { color: i % 2 ? COL.teal : COL.main } });
    s.addText(String(i + 1), { x: MX, y: y + 0.08, w: 0.6, h: 0.6, fontFace: F, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
    txt(s, t, { x: MX + 0.85, y, w: 11.2, h: 0.95, fontSize: 15, valign: "middle" });
  });
}

// 6–7. งานวิจัยที่เกี่ยวข้อง
const LIT1 = [["Fama (1970)", "ตลาดมีประสิทธิภาพ ราคาสะท้อนข้อมูลที่มีอยู่", "ตั้งความคาดหวังว่าสัญญาณอ่อน และใช้ตัวเทียบทางสถิติ (E0)"],
  ["Jegadeesh (1990); Lehmann (1990)", "การกลับตัวระยะสั้นของผลตอบแทน", "อธิบาย feature ผลตอบแทนย้อนหลังและตำแหน่งราคาปิดในกรอบวัน"],
  ["Wilder (1978); Appel (2005); Bollinger (2001)", "RSI, ATR, MACD, Bollinger Bands", "feature ตัวชี้วัดทางเทคนิครายวันและแท่ง 60 นาที"],
  ["le Cessie & van Houwelingen (1992)", "logistic regression แบบ ridge (L2)", "โมเดลหลักของทิศทางรอบเช้าและโมเดลราคา"],
  ["Breiman (1996)", "bagging ลดความแปรปรวนของโมเดล", "เฉลี่ย 20 โมเดลด้วย block bootstrap รายบล็อก 20 วัน"],
  ["Torrence & Compo (1998); Grinsted et al. (2004)", "การแปลงเวฟเล็ต Morlet และ wavelet coherence", "สร้างภาพ scalogram และวิเคราะห์ความสัมพันธ์ตามความถี่"],
  ["He et al. (2016); Huang et al. (2017); Simonyan & Zisserman (2015); Szegedy et al. (2016); Krizhevsky et al. (2012); Deng et al. (2009)", "CNN ที่ฝึกล่วงหน้าบน ImageNet", "ดึงลักษณะจากภาพ scalogram และ fine-tune (E10, E19, E20)"]];
const LIT2 = [["Chen & Guestrin (2016); Ke et al. (2017); Prokhorenkova et al. (2018); Breiman (2001)", "XGBoost, LightGBM, CatBoost, Random Forest", "โมเดลเปรียบเทียบ (EZOO, E15, E17b, E20)"],
  ["Tashman (2000)", "การทดสอบนอกตัวอย่างตามลำดับเวลา", "การทดสอบแบบเดินหน้า ฝึกขั้นต่ำ 120 วัน ทดสอบทีละ 5 วัน"],
  ["Efron (1979); Künsch (1989)", "bootstrap และ block bootstrap", "ช่วงความเชื่อมั่นแบบสุ่มทั้งวันพร้อมกัน"],
  ["Bailey et al. (2017)", "ความน่าจะเป็นของ backtest overfitting (PBO/CSCV)", "ตรวจความเสี่ยงจากการเลือกแบบที่ดีโดยบังเอิญ"],
  ["Gneiting & Raftery (2007)", "scoring rule ที่เหมาะสม", "ใช้ log loss เป็นตัวชี้วัดหลักของความน่าจะเป็น"],
  ["Sharpe (1966); สูตร XIRR", "ผลตอบแทนต่อความเสี่ยงและอัตราผลตอบแทนตามวันที่จริง", "ประเมินมูลค่าทางการเงินของสัญญาณ"]];
[[LIT1, "งานวิจัยที่เกี่ยวข้อง (1/2)"], [LIT2, "งานวิจัยที่เกี่ยวข้อง (2/2)"]].forEach(([rows, title]) => {
  const s = slide(title, "รายการอ้างอิงฉบับเต็มในหน้าอ้างอิง (APA)", "งานวิจัยที่เกี่ยวข้อง");
  table(s, [["งานวิจัย", "แนวคิดหลัก", "นำมาปรับใช้ในงานนี้"], ...rows], { y: 1.2, colW: [4.4, 3.6, 4.1], rowH: 0.62, fontSize: 12 });
});

// 8. แหล่งข้อมูล
{
  const s = slide("แหล่งข้อมูลและการตรวจคุณภาพ", "docs/02_ข้อมูลและfeatures.md · results/e18_data_validation.md · scripts/00_fetch_*.py", "ข้อมูล");
  card(s, MX, 1.2, 3.9, 3.0, "Settrade Open API", "ราคารายวันย้อนหลังราว 3 ปี (ราคาเปิดประมูลและราคาปิดทางการ) และแท่ง 1 นาทีย้อนหลังราว 4 วันทำการ · ใช้ในระบบส่งจริง");
  card(s, 4.7, 1.2, 3.9, 3.0, "yfinance", `แท่ง 60 นาทีเป็นข้อมูลหลักในการฝึก (${comma(D.data.n_days)} วันทำการ) · แท่ง 5 นาทีราว 60 วัน · ข้อมูลตลาดต่างประเทศ (E18)`, COL.teal);
  card(s, 8.8, 1.2, 3.9, 3.0, "การตรวจคุณภาพ", "ราคา 4 จุดของโจทย์จากแท่ง 60 นาทีตรงกับ Settrade 1 นาที 32/32 จุด · ใช้ราคาเปิดและราคาปิดทางการจาก Settrade");
  txt(s, "ไม่เผยแพร่ข้อมูลราคาดิบ เนื่องจากเงื่อนไขการใช้งานของแหล่งข้อมูลอนุญาตเฉพาะการใช้ส่วนบุคคล ตารางและกราฟในสไลด์นี้ใช้เพื่อการนำเสนอในรายวิชาเท่านั้น", { x: MX, y: 4.6, w: 12.1, h: 0.8, fontSize: 13, color: COL.muted });
}

// 9. การแบ่งข้อมูล
{
  const s = slide("การแบ่งข้อมูลตามเวลา", "dataset _wf · experiments/17_label_variants.py (WF_LAST_DAY) · results/e16_report.md", "ข้อมูล");
  const segs = [["ข้อมูลฝึกขั้นต่ำ 120 วัน", "24 ต.ค. 2566 –", 1.8, COL.muted], ["ทดสอบแบบเดินหน้า", "23 เม.ย. 2567 – 12 ต.ค. 2569", 6.3, COL.main], ["Holdout", "14–29 ต.ค. 2569", 1.6, COL.teal], ["ข้อมูลใหม่", "30 ต.ค. – 6 พ.ย. 2569", 1.2, "C8955F"], ["ส่งจริง", "9–13 พ.ย. 2569", 1.2, COL.dark]];
  let x = MX;
  segs.forEach(([h, d, w, c]) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.0, w: w - 0.05, h: 0.9, fill: { color: c }, line: { color: c } });
    s.addText(h, { x, y: 2.0, w: w - 0.05, h: 0.9, fontFace: F, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0.03 });
    s.addText(d, { x, y: 3.0, w: w - 0.05, h: 0.6, fontFace: F, fontSize: 11, color: COL.text, align: "center", isTextBox: true, margin: 0 });
    x += w;
  });
  const n = D.cls_wf;
  bullets(s, [`ทดสอบแบบเดินหน้า: รอบเช้า ${comma(n["morning|A-ud"].n)} รอบ และรอบบ่าย ${comma(n["afternoon|E15 MLP"].n)} รอบ (รอบบ่ายเริ่ม 24 ก.ย. 2567 หลังสะสมข้อมูลฝึกครบ 120 วัน)`,
    "Holdout ใช้รายงานครั้งเดียว ไม่ใช้เลือกโมเดล",
    "ข้อมูลใหม่ 30 ต.ค. – 6 พ.ย. 2569 ใช้ตรวจเชิงพรรณนา",
    "โมเดลที่ใช้ส่งจริงฝึกด้วยข้อมูล 24 ต.ค. 2566 – 6 พ.ย. 2569"], { x: MX, y: 4.0, w: 12.1, h: 2.4, fontSize: 14 });
}

// 10. สถิติเชิงพรรณนา
{
  const s = slide("สถิติเชิงพรรณนาของทิศทางราคา", "dataset _wf (รอบที่มีป้ายทั้งหมด) · results/e14/kbank_flat_rate.csv", "ข้อมูล");
  const order = [["KBANK", "morning"], ["KBANK", "afternoon"], ["ADVANC", "morning"], ["ADVANC", "afternoon"]];
  const dist = order.map(([a, b]) => D.data.dist.find((d) => d.symbol === a && d.session === b));
  const labels = order.map(([a, b]) => `${a} ${b === "morning" ? "เช้า" : "บ่าย"}`);
  s.addChart(pres.charts.BAR, ["up", "flat", "down"].map((k, i) => ({ name: ["ขึ้น", "เท่าเดิม", "ลง"][i], labels, values: dist.map((d) => +(d[k] * 100).toFixed(1)) })),
    { x: MX, y: 1.2, w: 7.4, h: 5.2, barDir: "bar", barGrouping: "percentStacked", chartColors: [COL.teal, "C8955F", COL.main], showValue: true, dataLabelPosition: "ctr", dataLabelColor: "FFFFFF", dataLabelFontSize: 10, dataLabelFormatCode: "0.0", valAxisLabelFormatCode: "0%", catAxisOrientation: "maxMin", showLegend: true, legendPos: "b", valGridLine: { color: "E8D9CB", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "สัดส่วนทิศทาง (%) แยกหุ้นและรอบ", ...chartText });
  const kf = D.data.kbank_flat;
  const kb = (sess, t) => kf.find((r) => r.session === sess && r.tick1 === (t === "True"));
  table(s, [["KBANK วันนิ่ง", "ช่อง 0.50 บาท", "ช่อง 1.00 บาท"],
    ["รอบเช้า", `${pct(kb("morning", "False").flat)} (n=${kb("morning", "False").n})`, `${pct(kb("morning", "True").flat)} (n=${kb("morning", "True").n})`],
    ["รอบบ่าย", `${pct(kb("afternoon", "False").flat)} (n=${kb("afternoon", "False").n})`, `${pct(kb("afternoon", "True").flat)} (n=${kb("afternoon", "True").n})`]], { x: 8.3, y: 1.35, w: 4.4, colW: [1.4, 1.5, 1.5], rowH: 0.42, fontSize: 11 });
  txt(s, "ขนาดการขยับเฉลี่ย (จำนวนช่องราคา)", { x: 8.3, y: 3.05, w: 4.4, h: 0.35, fontSize: 12, bold: true, color: COL.main });
  table(s, [["กลุ่ม", "ช่องราคาเฉลี่ย"], ...dist.map((d, i) => [labels[i], num(d.move_ticks, 2)])], { x: 8.3, y: 3.45, w: 4.4, colW: [2.4, 2.0], rowH: 0.36, fontSize: 11 });
  txt(s, "หลัง KBANK เปลี่ยนเป็นช่อง 1 บาท สัดส่วนวันนิ่งเพิ่มขึ้นทั้งสองรอบ", { x: 8.3, y: 5.4, w: 4.4, h: 0.8, fontSize: 12 });
}

// 11. ภาพรวมกระบวนการ
{
  const s = slide("ภาพรวมกระบวนการ", "experiments/ · src/forecast/ · results/decision_rules.md", "วิธีดำเนินการ");
  const steps = ["เก็บ" + NL + "ข้อมูล", "สร้าง" + NL + "feature", "โมเดล" + NL + "ทิศทาง/ราคา", "ทดสอบแบบ" + NL + "เดินหน้า", "ทดสอบ" + NL + "นัยสำคัญ", "เลือกตาม" + NL + "เกณฑ์", "ระบบส่ง" + NL + "อัตโนมัติ"];
  steps.forEach((t, i) => s.addShape(pres.shapes.CHEVRON, { x: MX + i * 1.74, y: 2.1, w: 1.85, h: 1.1, fill: { color: i % 2 ? COL.teal : COL.main }, line: { color: "FFFFFF", width: 1 } }));
  steps.forEach((t, i) => s.addText(t, { x: MX + i * 1.74 + 0.36, y: 2.1, w: 1.15, h: 1.1, fontFace: F, fontSize: 10.5, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 }));
  bullets(s, ["ทุกการทดลองเขียนเกณฑ์และ commit ก่อนรัน (pre-registration)",
    "ทุกค่าที่ใช้ทายมาจากข้อมูลที่จบก่อนจุดตัด (08:30 และ 12:35)",
    "เลือกพารามิเตอร์ feature และตัวแทนด้วยข้อมูลฝึกเท่านั้น (nested walk-forward)",
    "เปลี่ยนโมเดลส่งจริงเมื่อชนะค่าตั้งต้นอย่างมีนัยสำคัญตามเกณฑ์ใน results/decision_rules.md เท่านั้น"], { x: MX, y: 3.9, w: 12.1, h: 2.6, fontSize: 15 });
}

// 12–15. Feature
const F0 = [["f_ret_1d", "ln(C_d−1 / C_d−2) ราคาปิดแท่งสุดท้ายของวัน", "โมเมนตัมและการกลับตัวระยะสั้น", "ทั้งสอง"],
  ["f_ret_5d", "ln(C_d−1 / C_d−6)", "แนวโน้ม 1 สัปดาห์", "ทั้งสอง"],
  ["f_r_same_mean5 / 20", "ค่าเฉลี่ยผลตอบแทนของรอบเดียวกัน 5 / 20 วันก่อน", "แนวโน้มเฉพาะช่วงเวลาของวัน", "ทั้งสอง"],
  ["f_r_prev_afternoon", "ผลตอบแทนรอบบ่ายของวันก่อน", "ความต่อเนื่องระหว่างรอบ", "ทั้งสอง"],
  ["f_rv_5d", "ส่วนเบี่ยงเบนมาตรฐานของผลตอบแทนรายวัน 5 วัน", "ปรับความมั่นใจตามความเสี่ยง", "ทั้งสอง"],
  ["f_r_open_to_cut", "ln(ราคาล่าสุด ณ จุดตัด / ราคาเปิดเช้า)", "การเคลื่อนไหวของวันก่อนส่ง", "บ่าย"],
  ["f_gap_open", "ln(ราคาเปิดเช้า / ราคาปิดวันก่อน)", "ช่องว่างราคาเปิด", "บ่าย"],
  ["f_rv_intra", "√Σ(ผลตอบแทนแท่งภาคเช้า)²", "ความผันผวนภาคเช้า", "บ่าย"]];
{
  const s = slide("Feature หมวด 1: ผลตอบแทนและความผันผวน (F0)", "src/forecast/features/build.py (daily_features, intraday_features, build_features)", "วิธีดำเนินการ");
  txt(s, "รอบเช้าใช้ 6 ตัว และรอบบ่ายใช้ 9 ตัว (รวม 3 ตัวที่คำนวณจากข้อมูลภาคเช้าของวันนั้น) · C คือราคาปิดแท่งสุดท้ายของวัน", { x: MX, y: 1.1, w: 12.1, h: 0.45, fontSize: 13, color: COL.muted });
  table(s, [["ชื่อ", "วิธีคิด", "ช่วยอะไร", "รอบ"], ...F0], { y: 1.65, colW: [2.6, 4.9, 3.6, 1.0], rowH: 0.52, fontSize: 12 });
}
const FD = [["fd_ret_1, 2, 3, 5, 10", "(C_t / C_t−k) − 1 ตัดปลายแบบ expanding 1–99%", "โมเมนตัมหลายช่วงเวลา"],
  ["fd_hl_range", "(High − Low) / Close", "ความผันผวนภายในวัน"],
  ["fd_oc_change", "(Close − Open) / Open ตัดปลายแบบ expanding", "แรงซื้อขายระหว่างวัน"],
  ["fd_close_pos", "(Close − Low) / (High − Low) · ถ้า High = Low ใช้ 0.5", "ตำแหน่งราคาปิดในกรอบวัน จับการกลับตัว"],
  ["fd_c_sma5, fd_c_sma20", "Close / SMA5, Close / SMA20", "ระยะห่างจากค่าเฉลี่ยสั้นและกลาง"],
  ["fd_c_ema5, fd_c_ema20", "Close / EMA5, Close / EMA20", "ระยะห่างจากค่าเฉลี่ยถ่วงน้ำหนัก"],
  ["fd_sma5_sma20", "SMA5 / SMA20", "ทิศทางแนวโน้มสั้นเทียบกลาง"],
  ["fd_rsi14", "100 − 100/(1 + RS) · RS = ค่าเฉลี่ย Wilder ขาขึ้น/ขาลง 14 วัน", "ภาวะซื้อมาก/ขายมาก"]];
const FD2 = [["fd_macd, fd_macd_signal, fd_macd_hist", "(EMA12 − EMA26)/Close, เส้นสัญญาณ EMA9/Close, ส่วนต่าง/Close", "โมเมนตัมที่ปรับตามระดับราคา"],
  ["fd_bb_pos", "(Close − (SMA20 − 2σ20)) / (4σ20)", "ตำแหน่งในแถบ Bollinger"],
  ["fd_bb_width", "4σ20 / SMA20", "ความผันผวนสัมพัทธ์"],
  ["fd_atr14", "ค่าเฉลี่ย Wilder ของ True Range 14 วัน / Close", "ขนาดการขยับปกติ"],
  ["fd_up_streak", "จำนวนวันที่ปิดสูงขึ้นติดกัน", "การกลับตัวหลังขึ้นต่อเนื่อง"],
  ["fd_c_max20, fd_c_min20", "Close / Max20 − 1, Close / Min20 − 1", "ตำแหน่งในกรอบ 20 วัน"],
  ["fd_vol_ratio", "Volume / ค่าเฉลี่ย Volume 20 วัน", "ความผิดปกติของปริมาณซื้อขาย"]];
[[FD, "Feature หมวด 2: ตัวชี้วัดทางเทคนิครายวัน (F-daily, 1/2)"], [FD2, "Feature หมวด 2: ตัวชี้วัดทางเทคนิครายวัน (F-daily, 2/2)"]].forEach(([rows, title], k) => {
  const s = slide(title, "src/forecast/features/cn340.py (daily_features) · แท่งรายวัน Settrade ณ ราคาปิดวันทำการก่อนวันที่ทาย", "วิธีดำเนินการ");
  if (k === 0) txt(s, "24 ตัว ใช้ทั้งสองรอบ · ค่ามาจากแท่งรายวันของวันทำการก่อนวันที่ทาย จึงไม่มีข้อมูลของวันนั้นปน", { x: MX, y: 1.1, w: 12.1, h: 0.45, fontSize: 13, color: COL.muted });
  table(s, [["ชื่อ", "วิธีคิด", "ช่วยอะไร"], ...rows], { y: 1.65, colW: [3.3, 5.3, 3.5], rowH: 0.55, fontSize: 12 });
});
{
  const s = slide("Feature หมวด 3–6: วัน หุ้น วันนิ่ง และแท่ง 60 นาที", "src/forecast/features/dataset.py (add_flat_features) · cn340.py (bar_features) · outputs/models/R4ATC_UN/E15/mlp", "วิธีดำเนินการ");
  table(s, [["หมวด", "ชื่อ", "วิธีคิด", "ช่วยอะไร"],
    ["3 วันในสัปดาห์", "fl_dow_0…4", "one-hot ของวันจันทร์–ศุกร์", "ผลของวันในสัปดาห์"],
    ["4 ตัวแปรบอกหุ้น", "f_sym_KBANK / ADVANC", "dummy ของหุ้น", "แยกค่าเฉลี่ยของแต่ละหุ้นในโมเดลรวม"],
    ["5 วันนิ่ง", "fl_exp_ticks", "ค่าเฉลี่ย |r| ของรอบเดียวกัน 20 วัน × P(c) / tick_size(P(c))", "การขยับที่คาดเป็นจำนวนช่อง"],
    ["5 วันนิ่ง", "fl_prev_range_ticks", "(สูงสุด − ต่ำสุด) ของรอบล่าสุดก่อนจุดตัด / tick_size(P(c))", "ความกว้างรอบล่าสุดเป็นช่อง"],
    ["5 วันนิ่ง", "fl_flat20_dir / fl_flat20_price", "สัดส่วนวันนิ่ง 20 วันก่อน (เทียบราคาเปิด / เทียบ P(c))", "โอกาสราคาเท่าเดิม"],
    ["6 แท่ง 60 นาที (บ่าย)", "fh_rsi14, fh_macd_hist, fh_bb_pos, fh_c_sma20, fh_atr14", "สูตรเดียวกับหมวด 2 บนแท่ง 60 นาทีต่อเนื่องข้ามวันถึงจุดตัด", "สภาวะตลาดระยะสั้นก่อนส่ง"]], { y: 1.2, colW: [2.2, 3.4, 4.2, 2.3], rowH: 0.58, fontSize: 12 });
  txt(s, `E15 MLP รอบบ่ายเลือก ${D.mlp.features.length} feature ด้วย nested walk-forward ได้แก่ ${D.mlp.features.join(", ")}`, { x: MX, y: 5.4, w: 12.1, h: 1.0, fontSize: 12, color: COL.muted });
}
// 16. feature ที่ไม่ใช้ + importance
{
  const s = slide("Feature ที่ทดลองแต่ไม่ใช้ และความสำคัญของ feature", D.sources.importance, "วิธีดำเนินการ");
  card(s, MX, 1.2, 5.3, 1.6, "ชั้น 2 จากแท่ง 5 นาที (E13)", "ผลตอบแทน 15 นาทีสุดท้าย ราคาเทียบ TWAP และตำแหน่งในกรอบภาคเช้า · รันเงาเท่านั้น", COL.main, 12);
  card(s, MX, 2.95, 5.3, 1.75, "ตลาดต่างประเทศข้ามคืน (E18, E18b, E27)", "S&P 500, THD, Nikkei, KOSPI, USD/THB · fx_thd_prem = การเปลี่ยนแปลงของ THD − SET50 + ค่าเงิน · ทาย gap ได้แต่ไม่ช่วยทิศทาง", COL.teal, 12);
  card(s, MX, 4.85, 5.3, 1.55, "ภาพ scalogram 3 ช่อง", "ผลตอบแทน, |ผลตอบแทน| และ log volume จาก Morlet CWT · ไม่มี CNN ใดชนะตัวเทียบ", COL.main, 12);
  const imp = D.importance;
  s.addChart(pres.charts.BAR, [{ name: "Δ log loss", labels: imp.map((d) => d.feature), values: imp.map((d) => +(d.d_logloss * 1000).toFixed(2)) }],
    { x: 6.2, y: 1.15, w: 6.5, h: 5.3, barDir: "bar", chartColors: [COL.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: COL.text, dataLabelFormatCode: "0.00", catAxisOrientation: "maxMin", valGridLine: { color: "E8D9CB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Permutation importance ของ A-ud รอบเช้า (log loss ที่เพิ่ม × 1,000)", ...chartText });
}
// 17. โมเดลที่ใช้
{
  const s = slide("โมเดลที่ใช้ส่งจริง", "outputs/models/R4ATC/E12/fd_lr · outputs/models/R4ATC_UN/E15/mlp · configs/rules.yaml (live.models)", "วิธีดำเนินการ");
  card(s, MX, 1.2, 5.95, 1.55, "ตัวเทียบ", "E0-ud ทางที่เกิดบ่อยในวันที่ราคาขยับ · E0-un ทาย \"ไม่ขึ้น\" ทุกรอบ (สัดส่วนวันขึ้นในอดีต) · ราคาล่าสุด P(c)", COL.muted, 13);
  card(s, 6.75, 1.2, 5.95, 1.55, "โมเดลราคา PR-A", "A-price ทาย P(ราคาปลาย = P(c)) และ p_price ทาย P(สูงกว่า | ไม่เท่า) แล้วเลื่อน ±1 ช่องเมื่อความน่าจะเป็นเกิน 0.5 · ช่วงราคา 50%/80% จาก quantile ในอดีต", COL.teal, 13);
  card(s, MX, 2.95, 5.95, 3.45, "ทิศทางรอบเช้า: A-ud", `L2 logistic regression รวมสองหุ้น ใช้ feature หมวด 1–4 (${D.aud.n_features} คอลัมน์) · เลือก C จาก {0.003, 0.01, 0.03, 0.1} ด้วย inner time-series CV (โมเดลที่ล็อกได้ C = ${D.aud.C}) · bagging ${D.aud.n_bags} รอบ (block 20 วัน) · ฝึกแบบขึ้น/ลง ตัดวันนิ่ง · ทาย "ขึ้น" เมื่อ P ≥ 0.5 · เวอร์ชัน ${D.aud.version}`);
  card(s, 6.75, 2.95, 5.95, 3.45, "ทิศทางรอบบ่าย: E15 MLP", `MLP ของ scikit-learn ชั้นซ่อน ${D.mlp.hidden.join("×")} โหนด, alpha = ${num(D.mlp.alpha, 2)} จูนด้วย Optuna และเลือก ${D.mlp.features.length} feature ภายใน nested walk-forward · ฝึกแบบขึ้น/ไม่ขึ้น · ทาย "ขึ้น" เมื่อ P > 0.5 · เวอร์ชัน ${D.mlp.version} ฝึกถึง ${thDate(D.mlp.trained_until)}`, COL.teal);
}
// 18. โมเดลที่ทดลอง + ลำดับการพัฒนา
{
  const s = slide("ลำดับการพัฒนาและโมเดลที่ทดลองแต่ไม่ได้ใช้", "ประวัติ commit · docs/reports/report_round2–7.pdf · results/e14–e28_report.md", "วิธีดำเนินการ");
  table(s, [["ช่วง", "สิ่งที่เปลี่ยน", "ผลสำคัญ / บทเรียน"],
    ["รอบ 1", "ตั้งระบบ baseline, logistic, GBM และ CNN จากภาพ (E0–E8)", "สัญญาณอ่อนมาก จึงต้องมีตัวเทียบและนัยสำคัญ"],
    ["รอบ 2", "ตัวชี้วัดใหม่, E9, โมเดลวันนิ่ง (E9-FLAT), CNN fine-tune (E10)", "โมเดลวันนิ่งใช้กับราคา · CNN overfit เร็ว"],
    ["รอบ 3", "feature รายวัน (E12), ล็อกโมเดล, holdout, ระบบส่งจริง", "A-ud ชนะ E0-ud บนเป้าหมายขึ้น/ลง"],
    ["รอบ 4–5", "ป้าย ATC และราคาปิดทางการ, fallback, ร่างรอบเช้า, ชั้น 2 (E13)", "backtest ตรงกับสถานการณ์จริง"],
    ["รอบ 6 (ปฏิบัติการ)", "วัดความหน่วงข้อมูลและตั้งจุดตัด 08:30 / 12:35", "ส่งทันเส้นตายใหม่"],
    ["รอบ 7", "เป้าหมายขึ้น/ไม่ขึ้น (E16), 7 โมเดล + Optuna (E15), pretrain→fine-tune (E14), ตัดสิน 4 ช่อง", "รอบบ่ายเปลี่ยนเป็น E15 MLP"],
    ["หลังรอบ 7", "ลด overfit (E20), ablation (E21), ราคา (E22–E25), feature ใหม่ (E26), gap (E27), fine-tune แยกช่วง (E28)", "ไม่มีแบบใดชนะค่าตั้งต้นจนต้องเปลี่ยน"]], { y: 1.15, colW: [1.8, 5.7, 4.6], rowH: 0.5, fontSize: 11 });
  txt(s, "โมเดลที่ทดลองแต่ไม่ได้ใช้: LightGBM, XGBoost, CatBoost, Random Forest, SVM, ElasticNet, ensemble, CNN (AlexNet, VGG16-BN, ResNet18, DenseNet121, Inception-v3) บน scalogram แท่ง 60 นาที / 5 นาที / 1 นาที, pretrain→fine-tune และ feature ต่างประเทศ", { x: MX, y: 5.35, w: 12.1, h: 1.0, fontSize: 12, color: COL.muted });
}
// 19. การทดสอบ
{
  const s = slide("การทดสอบและการเลือกโมเดล", "src/forecast/evaluation/splits.py · tests/ · results/overfit_report.md", "วิธีดำเนินการ");
  for (let i = 0; i < 6; i++) {
    const y = 1.25 + i * 0.42;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 2.0 + i * 0.5, h: 0.3, fill: { color: COL.teal }, line: { color: COL.teal } });
    s.addShape(pres.shapes.RECTANGLE, { x: MX + 2.05 + i * 0.5, y, w: 0.5, h: 0.3, fill: { color: COL.main }, line: { color: COL.main } });
  }
  txt(s, "ฝึก (ขั้นต่ำ 120 วัน ขยายต่อเนื่อง)", { x: MX, y: 3.85, w: 3.5, h: 0.3, fontSize: 11, color: COL.teal, bold: true });
  txt(s, "ทดสอบทีละ 5 วัน", { x: 3.8, y: 3.85, w: 2.0, h: 0.3, fontSize: 11, color: COL.main, bold: true });
  bullets(s, ["Holdout 14–29 ต.ค. 2569 ใช้ครั้งเดียว ไม่ใช้เลือกโมเดล",
    "Pre-registration: เขียนเกณฑ์และ commit ก่อนรันทุกการทดลอง",
    "Nested walk-forward: เลือกพารามิเตอร์ feature และตัวแทนจากข้อมูลฝึกเท่านั้น",
    "ป้องกันข้อมูลรั่ว: test อัตโนมัติแบบตัดข้อมูลหลังจุดตัด (truncation) และเปลี่ยนค่าหลังจุดตัด (perturbation)",
    "ตรวจ overfit: เทียบ train gain กับ test gain และคำนวณ PBO/CSCV"], { x: 6.6, y: 1.2, w: 6.1, h: 5.2, fontSize: 14 });
}
// 20. วิธีทดสอบนัยสำคัญ
{
  const s = slide("วิธีทดสอบนัยสำคัญ", "experiments/31_e14_finetune.py (boot_ci, compare) · experiments/38_e18b_gap_shift.py (boot_rmse) · results/decision_rules.md", "วิธีดำเนินการ");
  SIG_PAGE = page;
  const items = [["1 ค่าที่เปรียบเทียบ", "ผลต่างระหว่างโมเดลกับตัวเทียบบนแถวเดียวกัน เช่น ความแม่นยำของโมเดล − ความแม่นยำของตัวเทียบ, log loss หรือ RMSE"],
    ["2 ช่วงความเชื่อมั่น", "รายงาน CI พร้อมระดับ: 95% สำหรับงานรายงาน · 98.33–99.77% เมื่อปรับ Bonferroni ตามจำนวนแบบ · 99.7% สำหรับการตัดสินเปลี่ยนโมเดล"],
    ["3 วิธีคำนวณ", "block bootstrap รายวัน 2,000 ครั้ง สุ่มทั้งวันพร้อมกันทุกแถวของวันนั้น · RMSE คำนวณใหม่ทุกรอบแบบจับคู่วันเดียวกัน"],
    ["4 เกณฑ์ตัดสิน", "มีนัยสำคัญเมื่อ CI ไม่ครอบศูนย์ · การเปลี่ยนโมเดลต้องดีกว่าทั้ง log loss และไม่แย่กว่าในหุ้นใดหุ้นหนึ่ง"],
    ["5 จำนวนตัวอย่าง", "ระบุ n ทุกตาราง · walk-forward รอบเช้า 1,196 และรอบบ่าย 989 รอบ (ทิศทาง)"]];
  items.forEach(([h, b], i) => card(s, MX + (i % 2) * 6.15, 1.2 + Math.floor(i / 2) * 1.75, i === 4 ? 12.1 : 5.95, 1.6, h, b, i % 2 ? COL.teal : COL.main, 12));
}
// 21. ปัญหา → วิธีแก้
{
  const s = slide("ปัญหาที่พบและวิธีแก้", "รายงานรอบ 2–7 · results/e16, e20, e27, e28 · docs/02_ข้อมูลและfeatures.md", "วิธีดำเนินการ");
  table(s, [["#", "วิธีแก้", "ผลหลังแก้"], ...PROBS.map((p) => [p[0], p[3], p[4]])], { y: 1.15, colW: [0.6, 6.4, 5.1], rowH: 0.43, fontSize: 11 });
}
// 22. ระบบส่งจริง
{
  const s = slide("ระบบส่งจริง", "scripts/install_cron.sh · scripts/05_predict_live.py · scripts/10_watchdog.py · docs/live_runbook.md", "วิธีดำเนินการ");
  const tl = [["17:30", "เก็บข้อมูลรายวัน · ร่างรอบเช้า · ฝึกชั้น 2"], ["08:30", "ทำนายรอบเช้า · เทียบกับร่าง"], ["08:50", "watchdog ตรวจไฟล์ส่ง"], ["12:35", "ทำนายรอบบ่าย"], ["12:50", "watchdog ตรวจไฟล์ส่ง"]];
  tl.forEach(([t, d], i) => {
    const x = MX + i * 2.45;
    s.addShape(pres.shapes.OVAL, { x: x + 0.75, y: 1.3, w: 0.7, h: 0.7, fill: { color: i % 2 ? COL.teal : COL.main }, line: { color: "FFFFFF" } });
    s.addText(t, { x, y: 2.1, w: 2.2, h: 0.4, fontFace: F, fontSize: 16, bold: true, color: COL.text, align: "center", isTextBox: true, margin: 0 });
    s.addText(d, { x, y: 2.5, w: 2.2, h: 0.9, fontFace: F, fontSize: 11, color: COL.text, align: "center", isTextBox: true, margin: 0 });
  });
  s.addShape(pres.shapes.LINE, { x: MX + 1.1, y: 1.65, w: 9.8, h: 0, line: { color: COL.line, width: 1.5 } });
  card(s, MX, 3.75, 3.9, 2.6, "เครื่องหลัก", "VM DigitalOcean (สิงคโปร์) ตั้งเวลาด้วย cron เขตเวลาไทย · ดึงข้อมูลอย่างเดียว ไม่มีโค้ดส่งคำสั่งซื้อขาย");
  card(s, 4.7, 3.75, 3.9, 2.6, "การแจ้งเตือนและทางสำรอง", "ส่งผลและคำเตือนผ่าน Telegram · ข้อมูลขาดใช้แหล่งสำรอง · รอบเช้าล้มใช้ร่างจากเย็นวันก่อน", COL.teal);
  card(s, 8.8, 3.75, 3.9, 2.6, "โน้ตบุ๊กสำรอง", "โค้ดชุดเดียวกัน · ห้าม login Settrade จากโน้ตบุ๊ก 08:25–08:55 และ 12:30–12:45 เพราะ Settrade ให้ login ทีละ session");
}

// 23. Confusion matrix
function cm(s, x, title, c, sub) {
  s.addText(title, { x, y: 1.15, w: 5.9, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: COL.main, isTextBox: true, margin: 0 });
  const cell = (v, tot, hi) => ({ text: `${comma(v)}\n(${pct(v / tot)})`, options: { align: "center", bold: true, fontSize: 14, color: hi ? "FFFFFF" : COL.text, fill: { color: hi ? (v / tot > 0.3 ? COL.main : "C46B80") : (v / tot > 0.3 ? "E8BBC6" : "F6E2D3") } } });
  const n = c.n;
  s.addTable([[{ text: "", options: { fill: { color: COL.bg } } }, { text: "ทาย ขึ้น", options: { bold: true, align: "center", fill: { color: COL.dark }, color: "FFFFFF" } }, { text: "ทาย ไม่ขึ้น", options: { bold: true, align: "center", fill: { color: COL.dark }, color: "FFFFFF" } }],
    [{ text: "จริง ขึ้น", options: { bold: true, valign: "middle", fill: { color: COL.dark }, color: "FFFFFF" } }, cell(c.tp, n, true), cell(c.fn, n, false)],
    [{ text: "จริง ไม่ขึ้น", options: { bold: true, valign: "middle", fill: { color: COL.dark }, color: "FFFFFF" } }, cell(c.fp, n, false), cell(c.tn, n, true)]],
    { x, y: 1.65, w: 5.9, colW: [1.5, 2.2, 2.2], rowH: [0.45, 1.05, 1.05], fontFace: F, fontSize: 12, border: { type: "solid", pt: 1, color: "FFFFFF" } });
  txt(s, sub, { x, y: 4.35, w: 5.9, h: 1.0, fontSize: 12 });
}
{
  const s = slide("Confusion matrix ของโมเดลที่ใช้จริง (walk-forward)", D.sources.cls, "ผลลัพธ์");
  const a = D.cls_wf["morning|A-ud"], m = D.cls_wf["afternoon|E15 MLP"];
  cm(s, MX, `รอบเช้า: A-ud (n = ${comma(a.n)})`, a, `ทาย "ขึ้น" ${pct(a.pred_up_rate)} ของรอบ ขณะที่วันขึ้นจริงมี ${pct(a.up_rate)} · วันนิ่งนับเป็น "ไม่ขึ้น"`);
  cm(s, 6.8, `รอบบ่าย: E15 MLP (n = ${comma(m.n)})`, m, `ทาย "ขึ้น" เพียง ${comma(m.tp + m.fp)} ครั้งจาก ${comma(m.n)} รอบ (${pct(m.pred_up_rate)}) ขณะที่วันขึ้นจริงมี ${pct(m.up_rate)}`);
  txt(s, "ชุดข้อมูล: walk-forward 23 เม.ย. 2567 – 12 ต.ค. 2569 · กติกาให้คะแนนจริง (ขึ้น/ไม่ขึ้น) · ผลแยกหุ้นอยู่ในภาคผนวก", { x: MX, y: 5.6, w: 12.1, h: 0.5, fontSize: 11, color: COL.muted });
}
// 24. ตัวชี้วัด
{
  const s = slide("ตัวชี้วัดการจำแนกทิศทาง", D.sources.cls + " · holdout: run_variant(holdout_eval=True)", "ผลลัพธ์");
  const cols = [["morning|A-ud", "เช้า A-ud"], ["morning|E0-un", "เช้า E0-un"], ["afternoon|E15 MLP", "บ่าย E15 MLP"], ["afternoon|E0-un", "บ่าย E0-un"]];
  const g = (k) => D.cls_wf[k];
  const rows = [["ตัวชี้วัด (walk-forward)", ...cols.map((c) => c[1])],
    ["Accuracy = (TP + TN) / N", ...cols.map((c) => pct(g(c[0]).accuracy))],
    ["Precision = TP / (TP + FP)", ...cols.map((c) => g(c[0]).precision === null ? "ไม่นิยาม*" : pct(g(c[0]).precision) + (c[0] === "afternoon|E15 MLP" ? "**" : ""))],
    ["Sensitivity = TP / (TP + FN)", ...cols.map((c) => pct(g(c[0]).recall))],
    ["Specificity = TN / (TN + FP)", ...cols.map((c) => pct(g(c[0]).specificity))],
    ["F1", ...cols.map((c) => g(c[0]).f1 === null ? "ไม่นิยาม*" : num(g(c[0]).f1, 3))],
    ["Balanced accuracy", ...cols.map((c) => pct(g(c[0]).balanced_accuracy))],
    ["AUC", ...cols.map((c) => num(g(c[0]).auc, 3))],
    ["Log loss", ...cols.map((c) => num(g(c[0]).logloss, 4))],
    ["n", ...cols.map((c) => comma(g(c[0]).n))]];
  table(s, rows, { y: 1.15, colW: [3.9, 2.05, 2.05, 2.05, 2.05], rowH: 0.38, fontSize: 12 });
  const h = D.cls_holdout;
  txt(s, `* E0-un ไม่ทาย "ขึ้น" เลย (TP + FP = 0) จึงหาค่า Precision และ F1 ไม่ได้ · ** E15 MLP ทาย "ขึ้น" เพียง 1 ครั้งและถูก จึงได้ 100% จากตัวอย่างเดียว\nHoldout (n = 20 ต่อรอบ): A-ud ${pct(h["morning|A-ud"].accuracy)} · E15 MLP ${pct(h["afternoon|E15 MLP"].accuracy)} · ความแม่นยำรอบบ่ายราว 65% มาจากอัตรา "ไม่ขึ้น" ที่สูง ไม่ใช่การแยกวันขึ้น เห็นได้จาก Sensitivity ${pct(g("afternoon|E15 MLP").recall)} และ Balanced accuracy ${pct(g("afternoon|E15 MLP").balanced_accuracy)}`, { x: MX, y: 5.1, w: 12.1, h: 1.5, fontSize: 11 });
}
// 25. นัยสำคัญ
{
  const s = slide("เทียบตัวเทียบพร้อมการทดสอบนัยสำคัญ", "experiments/60_make_slides.py (compare) · results/e16_report.md · results/e15_report.md · results/lock_decision.md", "ผลลัพธ์");
  const pick = (sess, a, b) => D.sig_wf.find((r) => r.session === sess && r.a === a && r.b === b && (r.group === "เช้า" || r.group === "บ่าย"));
  const rows = [["รอบ", "เทียบ", "n", "ความแม่นยำ a / b", "ผลต่างความแม่นยำ (จุด) [CI 99.7%]", "ผลต่าง log loss [CI 99.7%]", "ผล"]];
  [["morning", "A-ud", "E0-un"], ["afternoon", "E15 MLP", "E0-un"], ["afternoon", "E15 MLP", "A-ud"]].forEach(([sess, a, b]) => {
    const r = pick(sess, a, b);
    const sig = r.d_ll_hi < 0 || r.d_ll_lo > 0;
    rows.push([sess === "morning" ? "เช้า" : "บ่าย", `${a} − ${b}`, comma(r.n), `${pct(r.acc_a)} / ${pct(r.acc_b)}`, `${ppt(r.d_acc)} [${ppt(r.d_acc_lo)}, ${ppt(r.d_acc_hi)}]`, `${sgn(r.d_ll, 4)} ${ci(r.d_ll_lo, r.d_ll_hi, 4)}`,
      { text: sig ? (r.d_ll > 0 ? "แย่กว่าอย่างมีนัยสำคัญ" : "ดีกว่าอย่างมีนัยสำคัญ") : "ไม่มีนัยสำคัญ", options: { bold: true, color: sig ? COL.main : COL.text } }]);
  });
  table(s, rows, { y: 1.15, colW: [0.8, 2.0, 0.8, 2.0, 2.6, 2.4, 1.5], rowH: 0.5, fontSize: 11 });
  const u = D.aud_updown;
  card(s, MX, 3.35, 5.95, 2.9, "ไม่มีโมเดลใดชนะ E0-un", "E16 (เป้าหมายขึ้น/ไม่ขึ้น, CI 95%), E15 (7 โมเดล + Optuna, CI 99.55%) และ E14 (pretrain→fine-tune, CI 98.75%) ไม่มีแบบใดผ่าน · ดูหน้าถัดไปและภาคผนวก", COL.main, 13);
  card(s, 6.75, 3.35, 5.95, 2.9, "หลักฐานว่ามีสัญญาณเมื่อราคาขยับ", `บนเป้าหมายขึ้น/ลง (ไม่นับวันนิ่ง) A-ud แม่นยำ ${u.acc}% เทียบ E0-ud ${u.e0ud}% ผลต่าง ${u.diff.replace("-", "−")} จุด [${u.lo.replace("-", "−")}, ${u.hi.replace("-", "−")}] (CI 95%, walk-forward รอบ 3 ทั้งสองรอบ ป้ายชุดก่อนเปลี่ยนเป็น ATC · ไฟล์ไม่ได้บันทึก n) · ตัวเลขนี้ไม่ใช่กติกาให้คะแนนจริง`, COL.teal, 13);
  sigNote(s, 6.4);
}
// 26. ผลรวมทั้งโครงการ
{
  const s = slide("ผลรวมการทดลองทั้งโครงการ", "results/overfit_report.md · results/e14–e28_report.md · results/decision_4slots.md · results/e28/comparison_ledger.csv", "ผลลัพธ์");
  table(s, [["กลุ่ม", "คำถาม", "ผลดีที่สุดเทียบค่าตั้งต้น", "ผ่าน", "PBO เช้า / บ่าย"],
    ["รอบ 1–3 (E1–E12, EZOO)", "ทิศทางขึ้น/ลง", "A-ud +4.4 จุด [+1.5, +7.1] เทียบ E0-ud (CI 95%)", "เลือก A-ud", "0.43 / 0.17"],
    ["E10 CNN", "fine-tune CNN บน scalogram", "บ่าย E0 ดีที่สุด · overfit เร็ว 6/17 แบบ", "ไม่", "0.41 / 0.10"],
    ["E14", "pretrain → fine-tune", "ไม่มีแบบใดชนะ A-un (CI 98.75%)", "ไม่", "–"],
    ["E15", "7 โมเดล + Optuna + ensemble", "บ่าย MLP −0.0015 [−0.0079, +0.0054] เทียบ E0-un (CI 99.55%)", "ไม่ (เลือกเป็นช่องบ่าย)", "0.34 / 0.63"],
    ["E16", "เป้าหมายขึ้น/ไม่ขึ้น", "เช้า A-un −0.0045 [−0.0108, +0.0019] (CI 95%)", "ไม่", "–"],
    ["E17a / E17b", "โมเดลราคา", "0 จาก 44 แบบผ่าน (CI 99.77%)", "ไม่", "0.48 / 0.71"],
    ["E18 / E18b / E27", "ตลาดต่างประเทศและ gap", "ทาย gap ได้ (corr 0.29–0.35) แต่ RMSE ไม่ลด", "ไม่", "–"],
    ["E20", "ลด overfit", "SHR ผ่านเฉพาะทิศทางบ่ายเทียบ A-ud แต่ไม่ต่างจาก E15 MLP", "ไม่เปลี่ยน", "0.22 / 0.38"],
    ["E21–E25", "ablation, ราคา k, ราคา 2 ขั้น, 73 แบบ", "ไม่มีแบบใดผ่าน (CI 99.7%)", "ไม่", "–"],
    ["E26", "feature ใหม่ F1–F3", "F2 เช้าดีกว่า A-ud แต่ยังแพ้ E0-un (CI 98.33%)", "1 จาก 10", "–"],
    [`E28`, "fine-tune แยกช่วงเวลา", "เช้าแย่กว่า A-un +0.0116 [+0.0023, +0.0217] (CI 99.7%)", "ไม่", "0.10 / 0.46"]], { y: 1.15, colW: [2.3, 2.7, 4.5, 1.4, 1.2], rowH: 0.43, fontSize: 10.5 });
  txt(s, `จำนวนการเทียบที่ใช้ตัดสินสะสม ${D.ledger_total} ครั้ง (E13–E28) · PBO สูงกว่า 0.5 หมายถึงแบบที่ดีที่สุดในชุดฝึกมีโอกาสแย่กว่าค่ากลางเมื่อทดสอบ`, { x: MX, y: 6.45, w: 12.1, h: 0.4, fontSize: 10, color: COL.muted, italic: true });
}
// 27. ราคา
{
  const s = slide("ผลการทำนายราคา (walk-forward)", D.sources.price + " · " + D.price_e17.source, "ผลลัพธ์");
  const P = (m, g) => D.price.find((r) => r.model === m && r.group === g);
  const groups = ["เช้า", "KBANK เช้า", "ADVANC เช้า", "บ่าย", "KBANK บ่าย", "ADVANC บ่าย"];
  const rows = [["กลุ่ม", "n", "RMSE PR-A", "RMSE P(c)", "RMSE PR-mean", "MAE PR-A", "MAE P(c)", "PR-A − P(c) RMSE [CI 99.7%]"]];
  groups.forEach((g) => {
    const a = P("PR-A", g), c = P("P(c)", g), m = P("PR-mean", g);
    rows.push([g, comma(a.n), num(a.rmse, 3), num(c.rmse, 3), num(m.rmse, 3), num(a.mae, 3), num(c.mae, 3), `${sgn(a.d_rmse_pc, 3)} ${ci(a.d_rmse_pc_lo, a.d_rmse_pc_hi)}`]);
  });
  table(s, rows, { y: 1.15, colW: [1.7, 0.8, 1.35, 1.35, 1.5, 1.3, 1.3, 2.8], rowH: 0.4, fontSize: 11 });
  const e = D.price_e17;
  const cv = (g) => D.coverage.find((r) => r.group === g);
  txt(s, `แบบที่ดีที่สุดของ E17: เช้า ${e.morning.model} RMSE ${e.morning.rmse} (ผลต่าง ${sgn(e.morning.d_rmse)} ${ci(e.morning.lo, e.morning.hi)}) · บ่าย ${e.afternoon.model} RMSE ${e.afternoon.rmse} (${sgn(e.afternoon.d_rmse)} ${ci(e.afternoon.lo, e.afternoon.hi)}) เทียบ PR-A ที่ CI 99.7%`, { x: MX, y: 4.15, w: 12.1, h: 0.7, fontSize: 12 });
  table(s, [["ช่วงราคา PR-A", "Coverage 50%", "Coverage 80%", "ความกว้าง 50% (บาท)", "ความกว้าง 80% (บาท)"],
    ...["เช้า", "บ่าย"].map((g) => [g, pct(cv(g).cover50), pct(cv(g).cover80), num(cv(g).width50, 2), num(cv(g).width80, 2)])], { y: 4.95, colW: [2.5, 2.4, 2.4, 2.4, 2.4], rowH: 0.38, fontSize: 11 });
  sigNote(s, 6.25, " · RMSE เป็นตัวชี้วัดหลัก ไม่มีแบบใดต่างจาก P(c) อย่างมีนัยสำคัญ");
}
// 28. กราฟราคา
{
  const s = slide("ตัวอย่างราคาจริงเทียบราคาคาด และการกระจายของความคลาดเคลื่อน", "outputs/experiments/C1235 (PR-A walk-forward) · ภาพประกอบเท่านั้น", "ผลลัพธ์");
  const ex = D.price_example;
  s.addChart(pres.charts.LINE, [{ name: "ราคาจริง P(12:30)", labels: ex.map((d) => d.date), values: ex.map((d) => d.actual) }, { name: "ราคาคาด PR-A", labels: ex.map((d) => d.date), values: ex.map((d) => d.pred) },
    { name: "ขอบล่าง 80%", labels: ex.map((d) => d.date), values: ex.map((d) => d.lo80) }, { name: "ขอบบน 80%", labels: ex.map((d) => d.date), values: ex.map((d) => d.hi80) }],
    { x: MX, y: 1.15, w: 6.3, h: 5.2, chartColors: [COL.text, COL.main, "8FC1C1", "8FC1C1"], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5, showLegend: true, legendPos: "b", valGridLine: { color: "E8D9CB", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "KBANK รอบเช้า 15 วันสุดท้ายของ walk-forward (บาท)", valAxisMinVal: Math.floor(Math.min(...ex.map((d) => d.lo80)) - 2), ...chartText });
  const hb = D.err_hist;
  s.addChart(pres.charts.BAR, [{ name: "เช้า", labels: hb.bins.map(String), values: hb.morning }, { name: "บ่าย", labels: hb.bins.map(String), values: hb.afternoon }],
    { x: 7.1, y: 1.15, w: 5.6, h: 5.2, barGrouping: "clustered", chartColors: [COL.main, COL.teal], showLegend: true, legendPos: "b", valGridLine: { color: "E8D9CB", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "ความคลาดเคลื่อน PR-A − ราคาจริง (บาท, ตัดที่ ±10)", ...chartText });
}
// 29. เพดานและ XD
{
  const s = slide("ที่มาของความคลาดเคลื่อนราคา: gap และเงินปันผล", "results/e27/oracle.csv · results/e27/xd_effect.csv · results/e27_report.md", "ผลลัพธ์");
  const o = (m) => D.oracle.find((r) => r.model === m && r.group === "เช้า");
  const pa = o("PR-A"), op = o("P(10:00)");
  stat(s, MX, 1.25, 3.8, num(pa.rmse, 3), "RMSE รอบเช้าของ PR-A (บาท)");
  stat(s, 4.65, 1.25, 3.8, num(op.rmse, 3), "RMSE ถ้ารู้ราคาเปิด P(10:00) จริง (เพดานทางทฤษฎี)", COL.teal);
  const x = (m) => D.xd.find((r) => r.model === m);
  stat(s, 8.7, 1.25, 4.0, num(x("PR-A หักปันผลวัน XD").rmse_all, 3), "RMSE รอบเช้าเมื่อหักเงินปันผลวัน XD (ประมาณ)", "C8955F");
  bullets(s, [`ถ้ารู้ราคาเปิดจริง RMSE เปลี่ยน ${sgn(op.d_rmse, 3)} บาท ${ci(op.d_rmse_lo, op.d_rmse_hi)} และ MAE เปลี่ยน ${sgn(op.d_mae, 3)} ${ci(op.d_mae_lo, op.d_mae_hi)} เทียบ PR-A (CI 95%, n = ${comma(op.n)}) แต่ RMSE ยังเหลือ ${num(op.rmse, 3)} บาท ความคลาดเคลื่อนส่วนใหญ่จึงมาจากการขยับหลังตลาดเปิด`,
    `ระบบจริงหักเงินปันผลจาก P(c) ในวัน XD แต่ backtest ไม่ได้หัก · วัน XD มี 11 แถวในรอบเช้า แต่ทำให้ RMSE ของ backtest สูงกว่าระบบจริงราว ${num(pa.rmse - x("PR-A หักปันผลวัน XD").rmse_all, 2)} บาท`,
    "ราคาเปิดที่คาดจากตลาดต่างประเทศ (E18b) มี correlation กับ gap จริง 0.29–0.35 แต่ลด RMSE รวมได้ไม่มีนัยสำคัญ"], { x: MX, y: 3.15, w: 12.1, h: 3.2, fontSize: 14 });
}
// 30. การเงิน
{
  const s = slide("การประเมินทางการเงิน: XIRR และ Sharpe ratio", "results/financial/metrics.csv · results/financial_preregistration.md · ค่าธรรมเนียม settrade.com (broker 023) · อัตราดอกเบี้ยนโยบาย ธปท.", "ผลลัพธ์");
  const f = (p, ses, st, fee) => D.fin.find((r) => r["ช่วง"] === p && r["รอบ"] === ses && r["กลยุทธ์"] === st && r["ค่าธรรมเนียม"] === fee);
  const rows = [["กลยุทธ์ (walk-forward)", "ค่าธรรมเนียม", "XIRR", "Sharpe [CI 95%]", "ผลตอบแทนสะสม", "Max drawdown", "ครั้ง", "อัตราชนะ"]];
  for (const fee of ["0.0000%", "0.1680%"]) for (const [st, name] of [["S1", "S1 ซื้อเมื่อทายขึ้น"], ["B1", "B1 ซื้อทุกรอบ"], ["B2", "B2 ซื้อแล้วถือ"]]) {
    const r = f("walk-forward", "รวม", st, fee);
    rows.push([name, fee === "0.0000%" ? "ไม่มี" : "0.168%/ขา", pct(r.XIRR), `${num(r.Sharpe, 2)} ${ci(r.Sharpe_lo, r.Sharpe_hi, 2)}`, pct(r["ผลตอบแทนสะสม"]), pct(r.max_drawdown), comma(r["ครั้งที่ซื้อขาย"]), r["อัตราชนะ"] === null ? "–" : pct(r["อัตราชนะ"])]);
  }
  table(s, rows, { y: 1.15, colW: [2.4, 1.3, 1.1, 2.2, 1.5, 1.4, 0.9, 1.3], rowH: 0.38, fontSize: 11 });
  const dfx = D.fin_diff.find((r) => r["ช่วง"] === "walk-forward" && r["รอบ"] === "รวม" && r["ค่าธรรมเนียม"] === "0.0000%");
  bullets(s, [`สมมติฐาน: ลงทุน 100,000 บาทต่อหุ้นต่อรอบ ไม่ทบต้น · ซื้อที่ราคาต้นรอบ ขายที่ราคาปลายรอบ · ค่าธรรมเนียม 0.157% + VAT 7% = 0.168% ต่อขา · rf = 1.00% ต่อปี (อัตราดอกเบี้ยนโยบาย ธปท.) · ไม่คิด slippage และภาษีขาย`,
    `ไม่มีค่าธรรมเนียม: Sharpe ของ S1 สูงกว่า B1 ${sgn(dfx.Sharpe_S1_ลบ_B1, 2)} ${ci(dfx.lo, dfx.hi, 2)} (CI 95%, n = ${dfx["n_วัน"]} วัน) ไม่มีนัยสำคัญ · กำไรเกือบทั้งหมดมาจากรอบเช้า`,
    "มีค่าธรรมเนียม: ทุกกลยุทธ์ที่ซื้อขายทุกวันขาดทุน เพราะการขยับเฉลี่ยต่อรอบเล็กกว่าค่าธรรมเนียมไป-กลับ · รอบบ่าย E15 MLP ซื้อเพียง 1 ครั้ง จึงเท่ากับถือเงินสด",
    "Max drawdown คิดเป็นร้อยละของเงินลงทุนต่อช่อง (ไม่ทบต้น) จึงเกิน 100% ได้ · เป็นการจำลองย้อนหลัง ไม่ใช่คำแนะนำการลงทุน และไม่มีการส่งคำสั่งซื้อขายจริง"], { x: MX, y: 3.95, w: 12.1, h: 2.9, fontSize: 12 });
}
// 31. equity
{
  const s = slide("ผลตอบแทนสะสม: S1 เทียบ B1 และ B2 (walk-forward)", "results/financial/equity_wf_fee0.csv, equity_wf_feeon.csv", "ผลลัพธ์");
  ["0", "on"].forEach((fee, i) => {
    const e = D[`equity_fee${fee}`];
    s.addChart(pres.charts.LINE, [["S1", "S1 ซื้อเมื่อทายขึ้น"], ["B1", "B1 ซื้อทุกรอบ"], ["B2", "B2 ซื้อแล้วถือ"]].map(([k, n]) => ({ name: n, labels: e.map((r) => r.date.slice(0, 7)), values: e.map((r) => +(r[k] * 100).toFixed(2)) })),
      { x: MX + i * 6.15, y: 1.15, w: 5.95, h: 4.9, chartColors: [COL.main, COL.teal, "C8955F"], lineSize: 2, lineDataSymbol: "none", showLegend: true, legendPos: "b", valGridLine: { color: "E8D9CB", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: fee === "0" ? "ไม่มีค่าธรรมเนียม (% ของเงินลงทุน)" : "ค่าธรรมเนียม 0.168% ต่อขา (% ของเงินลงทุน)", catAxisLabelFrequency: 6, ...chartText });
  });
  txt(s, "S1 และ B1 คิดเป็นร้อยละของเงินลงทุนรวม 400,000 บาท (4 ช่อง) · B2 คิดจากเงินลงทุน 200,000 บาท (2 หุ้น) และถือข้ามคืน จึงได้รับผลจากแนวโน้มราคาทั้งช่วง", { x: MX, y: 6.15, w: 12.1, h: 0.6, fontSize: 11, color: COL.muted });
}
// 32. ส่งจริง
{
  const s = slide("ผลส่งจริง 9–13 พฤศจิกายน 2569", D.sources.live + ` · อัปเดตถึง ${thDate(D.live.last_day)}`, "ผลลัพธ์");
  const days = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"];
  const rows = [["วันที่", "รอบ", "KBANK ทาย / จริง", "KBANK ราคาคาด / จริง", "ADVANC ทาย / จริง", "ADVANC ราคาคาด / จริง"]];
  const mark = (q) => (q.correct === null ? "" : q.correct ? " ✓" : " ✗");
  for (const d of days) for (const ses of ["morning", "afternoon"]) {
    const k = D.live.rows.find((r) => r.date === d && r.session === ses && r.symbol === "KBANK");
    const a = D.live.rows.find((r) => r.date === d && r.session === ses && r.symbol === "ADVANC");
    const act = (v) => (v === null || v === undefined ? "รอผล" : v === "ขึ้น" ? "ขึ้น" : "ไม่ขึ้น");
    const dir = (q) => (q ? `${q.pred} / ${act(q.actual)}${mark(q)}` : "รอข้อมูล");
    const pr = (q) => (q ? `${num(q.point, 2)} / ${q.p_end === null ? "–" : num(q.p_end, 2)}` : "รอข้อมูล");
    rows.push([thDate(d).replace(" 2569", ""), ses === "morning" ? "เช้า" : "บ่าย", dir(k), pr(k), dir(a), pr(a)]);
  }
  table(s, rows, { y: 1.1, colW: [1.2, 0.8, 2.6, 2.4, 2.6, 2.5], rowH: 0.34, fontSize: 10.5 });
  const sm = D.live.summary, ex = D.live.expected;
  const sumrow = (k, n) => `${n}: ถูก ${sm[k].correct}/${sm[k].n} (${pct(sm[k].acc, 0)}) · RMSE ${num(sm[k].rmse, 2)} · MAE ${num(sm[k].mae, 2)} บาท`;
  txt(s, [sumrow("morning", "เช้า"), sumrow("afternoon", "บ่าย"), sumrow("all", "รวม")].join("\n") + `\nbacktest คาดความแม่นยำ ${pct(ex.morning.backtest_acc)} (เช้า) และ ${pct(ex.afternoon.backtest_acc)} (บ่าย) · ถ้าครบ 10 คำทายต่อรอบ ช่วงที่คาดราว ${num(ex.morning.range95[0], 1)}–${num(ex.morning.range95[1], 1)} และ ${num(ex.afternoon.range95[0], 1)}–${num(ex.afternoon.range95[1], 1)} ครั้ง (ประมาณแบบทวินาม 95%) · ตัวอย่างยังน้อยเกินสรุป`, { x: MX, y: 5.0, w: 12.1, h: 1.9, fontSize: 11 });
}
// 33–34. อภิปราย
{
  const s = slide("อภิปรายผลและข้อจำกัด", "results/e16_report.md · results/e16_morning_daily/summary.csv · results/e14_report.md · results/e27_report.md", "อภิปราย");
  const r32 = (m) => D.recent32.find((r) => r.model === m);
  card(s, MX, 1.2, 5.95, 2.5, "ไม่พบฝีมือเหนือสถิติพื้นฐาน", `ภายใต้กติกาขึ้น/ไม่ขึ้น ไม่มีโมเดลใดชนะ E0-un อย่างมีนัยสำคัญ (หน้า ${SIG_PAGE + 5}) และโมเดลเป้าหมายขึ้น/ไม่ขึ้นมักทาย "ไม่ขึ้น" เกือบทุกรอบ`, COL.main, 13);
  card(s, 6.75, 1.2, 5.95, 2.5, "เป้าหมายฝึกไม่ตรงกับวิธีให้คะแนน (A1)", `รอบเช้าคง A-ud ตามกติกาที่ลงทะเบียน · 32 รอบล่าสุด (14 ต.ค. – 6 พ.ย.) A-ud ถูก ${r32("A-ud").n_correct} รอบ เทียบ A-un ${r32("A-un").n_correct} รอบ (รายงานเชิงพรรณนา)`, COL.teal, 13);
  card(s, MX, 3.9, 3.9, 2.45, "ข้อมูล 1 นาทีไม่พอ", "ข้อมูลย้อนหลังมีราว 10 วัน ต้องการอย่างน้อย 1 ปี (เช่น SETSMART หรือ MT5) เพื่อทดสอบ CNN อย่างเป็นธรรม", COL.main, 12);
  card(s, 4.7, 3.9, 3.9, 2.45, "KBANK ช่วงช่อง 1 บาทสั้น", "แถวทดสอบช่วงนี้มี 95 วันต่อรอบ · C′ / C-adv รอบเช้าแม่นยำกว่า A-un +3.2 จุด แต่ CI กว้าง [−4.2, +10.5] (CI 98.75%)", COL.teal, 12);
  card(s, 8.8, 3.9, 3.9, 2.45, "backtest ราคาไม่หัก XD", "RMSE รอบเช้าใน backtest สูงกว่าระบบจริงราว 0.10 บาท ซึ่งมากกว่าผลต่างระหว่างโมเดลราคาที่ทดลอง", COL.main, 12);
}
{
  const s = slide("งานต่อยอด", "results/e19_report.md · results/e27_report.md · results/e28_report.md", "อภิปราย");
  const items = [["เก็บข้อมูล 1 นาทีย้อนหลังอย่างน้อย 1 ปี", "ทดสอบ CNN และ feature ระหว่างวันด้วย walk-forward ที่มีวันทดสอบพอ"],
    ["ทดสอบ C-adv ด้วยข้อมูลที่สะสม", "KBANK ช่อง 1 บาทจะมีข้อมูลมากขึ้นทุกวัน จึงวัดผลได้แม่นขึ้น"],
    ["รัน backtest ราคาใหม่โดยหักเงินปันผล", "ให้ RMSE ใน backtest ตรงกับระบบจริง ก่อนเทียบโมเดลราคาอีกครั้ง"],
    ["ยืนยันผลที่น่าสนใจด้วยข้อมูลใหม่", "F2 กับ A-ud รอบเช้า (E26) และราคา 2 ขั้น (E24) ควรทดสอบกับข้อมูลหลัง 13 พ.ย. 2569"]];
  items.forEach(([h, b], i) => card(s, MX + (i % 2) * 6.15, 1.25 + Math.floor(i / 2) * 2.6, 5.95, 2.35, h, b, i % 2 ? COL.teal : COL.main, 14));
}
// 35. สรุป
{
  const s = slide("สรุป", "สไลด์ก่อนหน้าและ results/", "สรุป");
  const a = D.cls_wf["morning|A-ud"], m = D.cls_wf["afternoon|E15 MLP"];
  const pts = [`สร้างระบบส่งอัตโนมัติบนคลาวด์ที่ส่งได้ทุกรอบในสัปดาห์ส่งจริง (${D.live.days} วันแรก) พร้อมแจ้งเตือนและทางสำรอง`,
    `ภายใต้กติกาขึ้น/ไม่ขึ้น ไม่มีโมเดลใดชนะการทาย "ไม่ขึ้น" ทุกรอบอย่างมีนัยสำคัญ (CI 99.7%) · A-ud รอบเช้าแม่นยำ ${pct(a.accuracy)} และ E15 MLP รอบบ่าย ${pct(m.accuracy)}`,
    "A-ud มีสัญญาณแยกทิศทางเมื่อราคาขยับ (+4.4 จุดเทียบ E0-ud, CI 95%) แต่ไม่พอชดเชยวันนิ่งที่นับเป็นไม่ขึ้น",
    "ราคาทุกแบบใกล้เคียงการใช้ราคาล่าสุด ความคลาดเคลื่อนส่วนใหญ่มาจากการขยับหลังตลาดเปิด",
    `ข้อมูลความละเอียดสูง CNN และ fine-tune ไม่ช่วยในข้อมูลชุดนี้ · ใช้ pre-registration และนับการเทียบสะสม ${D.ledger_total} ครั้ง เพื่อไม่ให้ผลดูดีเกินจริง`];
  pts.forEach((t, i) => {
    const y = 1.2 + i * 1.08;
    s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.12, w: 0.5, h: 0.5, fill: { color: i % 2 ? COL.teal : COL.main }, line: { color: i % 2 ? COL.teal : COL.main } });
    txt(s, t, { x: MX + 0.75, y, w: 11.3, h: 0.95, fontSize: 15, valign: "middle" });
  });
}
// 36–37. อ้างอิง
const REFS = ["Appel, G. (2005). Technical analysis: Power tools for active investors. FT Prentice Hall.",
  "Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. Journal of Computational Finance, 20(4), 39–69.",
  "Bollinger, J. (2001). Bollinger on Bollinger bands. McGraw-Hill.",
  "Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123–140.",
  "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
  "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
  "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. IEEE Conference on Computer Vision and Pattern Recognition, 248–255.",
  "Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1), 1–26.",
  "Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. The Journal of Finance, 25(2), 383–417.",
  "Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477), 359–378.",
  "Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. Nonlinear Processes in Geophysics, 11(5/6), 561–566.",
  "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. IEEE Conference on Computer Vision and Pattern Recognition, 770–778.",
  "Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. IEEE Conference on Computer Vision and Pattern Recognition, 4700–4708.",
  "Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. The Journal of Finance, 45(3), 881–898.",
  "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.",
  "Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25.",
  "Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. The Annals of Statistics, 17(3), 1217–1241.",
  "le Cessie, S., & van Houwelingen, J. C. (1992). Ridge estimators in logistic regression. Journal of the Royal Statistical Society: Series C (Applied Statistics), 41(1), 191–201.",
  "Lehmann, B. N. (1990). Fads, martingales, and market efficiency. The Quarterly Journal of Economics, 105(1), 1–28.",
  "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31.",
  "Sharpe, W. F. (1966). Mutual fund performance. The Journal of Business, 39(1), 119–138.",
  "Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. International Conference on Learning Representations.",
  "Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. IEEE Conference on Computer Vision and Pattern Recognition, 2818–2826.",
  "Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. International Journal of Forecasting, 16(4), 437–450.",
  "Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. Bulletin of the American Meteorological Society, 79(1), 61–78.",
  "Wilder, J. W. (1978). New concepts in technical trading systems. Trend Research.",
  "ตลาดหลักทรัพย์แห่งประเทศไทย. (2567). ประกาศเปลี่ยนเวลาเปิดภาคบ่าย มีผล 25 มีนาคม 2567. set.or.th",
  "ธนาคารแห่งประเทศไทย. (2569). ผลการประชุมคณะกรรมการนโยบายการเงิน: คงอัตราดอกเบี้ยนโยบายที่ร้อยละ 1.00 ต่อปี."];
[REFS.slice(0, 14), REFS.slice(14)].forEach((part, i) => {
  const s = slide(`อ้างอิง (${i + 1}/2)`, "รูปแบบ APA", "อ้างอิง");
  s.addText(part.map((t, j) => ({ text: t, options: { breakLine: j < part.length - 1, paraSpaceAfter: 4, indentLevel: 0 } })), { x: MX, y: 1.15, w: 12.1, h: 5.7, fontFace: F, fontSize: 11, color: COL.text, isTextBox: true, valign: "top", margin: 0.05 });
});

// ======================== ภาคผนวก ========================
{
  const s = slide("ภาคผนวก ก: ตัวชี้วัดแยกหุ้น (walk-forward)", D.sources.cls, "ภาคผนวก");
  const rows = [["รอบ · หุ้น · โมเดล", "n", "TP", "FP", "FN", "TN", "Accuracy", "Precision", "Sensitivity", "Specificity", "Balanced acc.", "Log loss"]];
  for (const [sess, model] of [["morning", "A-ud"], ["morning", "E0-un"], ["afternoon", "E15 MLP"], ["afternoon", "E0-un"]]) for (const sym of ["KBANK", "ADVANC"]) {
    const c = D.cls_wf[`${sess}|${model}|${sym}`];
    rows.push([`${sess === "morning" ? "เช้า" : "บ่าย"} · ${sym} · ${model}`, comma(c.n), c.tp, c.fp, c.fn, c.tn, pct(c.accuracy), pct(c.precision), pct(c.recall), pct(c.specificity), pct(c.balanced_accuracy), num(c.logloss, 4)]);
  }
  table(s, rows, { y: 1.15, colW: [2.9, 0.6, 0.55, 0.55, 0.55, 0.55, 1.0, 1.0, 1.05, 1.05, 1.1, 0.95], rowH: 0.42, fontSize: 10 });
  txt(s, "\"ไม่นิยาม\" หมายถึงไม่มีการทาย \"ขึ้น\" ในกลุ่มนั้น (TP + FP = 0)", { x: MX, y: 5.2, w: 12.1, h: 0.4, fontSize: 10, color: COL.muted });
}
{
  const s = slide("ภาคผนวก ข: ผล CNN fine-tune ทุกแบบ (E20 รอบ 2)", "results/e20/cnnb_finetune.csv · gain = log loss ของ E0 − log loss ของโมเดล (บวก = ดีกว่า) · CI 95%", "ภาคผนวก");
  const rows = [["โมเดล", "เช้า gain [CI 95%]", "บ่าย gain [CI 95%]", "epoch ดีสุด (เช้า / บ่าย)"]];
  const models = [...new Set(D.cnn.map((r) => r.model))];
  for (const mdl of models) {
    const a = D.cnn.find((r) => r.model === mdl && r.session === "morning"), b = D.cnn.find((r) => r.model === mdl && r.session === "afternoon");
    rows.push([mdl, `${sgn(a.test_gain, 4)} ${ci(a.test_lo, a.test_hi, 4)}`, `${sgn(b.test_gain, 4)} ${ci(b.test_lo, b.test_hi, 4)}`, `${a.best_epoch_median ?? "–"} / ${b.best_epoch_median ?? "–"}`]);
  }
  table(s, rows, { y: 1.1, colW: [3.0, 3.6, 3.6, 1.9], rowH: 0.3, fontSize: 9.5 });
}
{
  const s = slide("ภาคผนวก ค: PBO และ E28", "results/overfit_report.md · results/e20/pbo.csv · results/e28_report.md · results/e28/comparison_ledger.csv", "ภาคผนวก");
  table(s, [["กลุ่ม", "จำนวนแบบ", "PBO เช้า", "PBO บ่าย"], ["รอบ 1–3 รวม", "62", "0.43", "0.17"], ["E9", "–", "0.98", "0.71"], ["E10 CNN", "17", "0.41", "0.10"], ["E12", "–", "0.44", "0.44"], ["E15", "11", "0.34", "0.63"], ["ทิศทางรอบ 7 รวม", "20 / 19", "0.36", "0.73"], ["ราคารอบ 7 รวม", "14", "0.50", "0.90"], ["E20 ทิศทาง", "–", "0.22", "0.38"], ["E28", "7", "0.10", "0.46"]],
    { x: MX, y: 1.15, w: 5.6, colW: [2.4, 1.2, 1.0, 1.0], rowH: 0.4, fontSize: 11 });
  const v = D.e28_verdict;
  card(s, 6.6, 1.15, 6.1, 4.5, "E28: fine-tune แยกช่วงเวลา (feedback ข้อ 5)", v.map((r) => `${r["รอบ"]}: เทียบ A-un ${sgn(r["d_ll_A-un"], 4)} ${ci(r["lo_A-un"], r["hi_A-un"], 4)} · เทียบ E0-un ${sgn(r["d_ll_E0-un"], 4)} ${ci(r["lo_E0-un"], r["hi_E0-un"], 4)} · ${r["ผ่าน"] ? "ผ่าน" : "ไม่ผ่าน"}`).join("\n") + `\n\nผลต่าง log loss ที่ CI 99.7% · pretrain 70% แรก เว้น 5 วัน แล้ว fine-tune แยกรอบด้วยช่วงท้าย · ถ้าปรับ Bonferroni ตามการเทียบสะสม ${D.ledger_total} ครั้ง (CI 99.982%) ทุกช่องครอบศูนย์`, COL.teal, 12);
}
{
  const s = slide("ภาคผนวก ง: ไฟล์อ้างอิงใน repo", "github.com/hansera1/CN340-Final-StockPrediction", "ภาคผนวก");
  table(s, [["เนื้อหา", "ไฟล์"], ["กติกาและเวลา", "configs/rules.yaml · docs/01_โจทย์และกติกา.md"], ["feature", "src/forecast/features/ (build.py, cn340.py, dataset.py)"], ["เกณฑ์ตัดสินและการเลือกโมเดล", "results/decision_rules.md · results/decision_4slots.md · results/lock_decision.md"],
    ["ผลการทดลอง", "results/e14–e28_report.md และโฟลเดอร์ results/e14–e28/"], ["overfit และ PBO", "results/overfit_report.md"], ["การเงิน", "results/financial_preregistration.md · results/financial/ · experiments/61_financial_eval.py"],
    ["ระบบส่งจริง", "scripts/05_predict_live.py · scripts/10_watchdog.py · scripts/install_cron.sh · docs/live_runbook.md"], ["สร้างสไลด์", "experiments/60_make_slides.py · docs/slides/build_deck.js"]], { y: 1.2, colW: [3.4, 8.7], rowH: 0.5, fontSize: 12 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL_THEME) {
    const { applyTheme } = require(SKILL_THEME);
    await applyTheme(OUT, THEME);
  }
  // ฟอนต์ complex script (ภาษาไทย) ของธีม = Prompt
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT));
  const part = "ppt/theme/theme1.xml";
  let xml = await zip.file(part).async("string");
  xml = xml.replace(/<a:cs typeface="[^"]*"\s*\/>/g, `<a:cs typeface="${F}"/>`).replace(/<a:latin typeface="[^"]*"([^/]*)\/>/g, `<a:latin typeface="${F}"$1/>`);
  zip.file(part, xml);
  for (const k of Object.keys(zip.files).filter((k) => /^ppt\/charts\/chart\d+\.xml$/.test(k))) {
    let c = await zip.file(k).async("string");
    c = c.replace(/<a:latin typeface="Prompt"\s*\/>(\s*<a:ea [^>]*\/>)?(\s*<a:cs [^>]*\/>)?/g, '<a:latin typeface="Prompt"/><a:ea typeface="Prompt"/><a:cs typeface="Prompt"/>');
    zip.file(k, c);
  }
  const flat = new JSZip();
  for (const k of Object.keys(zip.files)) if (!zip.files[k].dir) flat.file(k, await zip.file(k).async("nodebuffer"), { createFolders: false });
  fs.writeFileSync(OUT, await flat.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("→", OUT, "·", page, "สไลด์ · หน้าวิธีทดสอบนัยสำคัญ =", SIG_PAGE);
})();
