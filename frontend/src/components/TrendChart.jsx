import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const MAX_POINTS = 50;
const COLORS = { line: "#a78bfa", grid: "#26262b", axis: "#75757e" };

export default function TrendChart({ history }) {
  const recent = history.slice(0, MAX_POINTS).reverse();
  const offset = history.length - recent.length;
  const data = recent.map((h, i) => ({
    n: offset + i + 1,
    positive: Math.round((h.probabilities.positive ?? 0) * 1000) / 10,
    text: h.text,
  }));
  const tick = { fill: COLORS.axis, fontSize: 12 };

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis dataKey="n" tick={tick} tickLine={false} axisLine={false} minTickGap={20} />
        <YAxis
          domain={[0, 100]}
          ticks={[0, 50, 100]}
          tickFormatter={(v) => `${v}%`}
          tick={tick}
          tickLine={false}
          axisLine={false}
        />
        <ReferenceLine y={50} stroke={COLORS.axis} strokeDasharray="3 3" />
        <Tooltip content={<TrendTooltip />} cursor={{ stroke: COLORS.grid }} isAnimationActive={false} />
        <Line
          type="linear"
          dataKey="positive"
          stroke={COLORS.line}
          strokeWidth={2}
          dot={data.length <= 20 ? { r: 2.5, fill: COLORS.line, strokeWidth: 0 } : false}
          activeDot={{ r: 4, strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function TrendTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div>
        #{point.n} · {point.positive}% positive
      </div>
      <div className="chart-tooltip-text">{point.text}</div>
    </div>
  );
}
