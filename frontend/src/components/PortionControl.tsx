import { Minus, Plus } from "lucide-react";
import { servingText } from "../lib/nutrition";
import type { Food } from "../lib/types";

type Props = {
  food: Food;
  servingLabel: string;
  quantity: number;
  onChange: (servingLabel: string, quantity: number) => void;
  idPrefix?: string;
};

export const defaultQty = (label: string) => (label === "gram" ? 100 : 1);

export default function PortionControl({ food, servingLabel, quantity, onChange, idPrefix = "p" }: Props) {
  const isGram = servingLabel === "gram";
  const step = isGram ? 10 : 0.5;
  const min = isGram ? 5 : 0.5;
  const set = (q: number) => onChange(servingLabel, Math.max(min, Math.round(q * 100) / 100));
  return (
    <div className="portion">
      <label className="field portion-serving" htmlFor={`${idPrefix}-serving`}>
        <span>Serving</span>
        <select id={`${idPrefix}-serving`} className="select" value={servingLabel}
          onChange={(e) => onChange(e.target.value, defaultQty(e.target.value))}>
          {food.servings.map((s) => (
            <option key={s.label} value={s.label}>
              {s.label === "gram" ? "grams" : `${s.label} (${s.grams} g)`}
            </option>
          ))}
        </select>
      </label>
      <div className="field">
        <span id={`${idPrefix}-qty-label`}>{isGram ? "Grams" : "How many"}</span>
        <div className="stepper" role="group" aria-labelledby={`${idPrefix}-qty-label`}>
          <button type="button" className="btn icon-btn" aria-label="Less" onClick={() => set(quantity - step)} disabled={quantity <= min}>
            <Minus aria-hidden="true" />
          </button>
          <input className="input num" inputMode="decimal" aria-label={`Quantity in ${servingText(servingLabel)}`}
            value={quantity} data-testid="qty"
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v) && v > 0 && v <= 2000) onChange(servingLabel, v);
            }} />
          <button type="button" className="btn icon-btn" aria-label="More" onClick={() => set(quantity + step)}>
            <Plus aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  );
}
