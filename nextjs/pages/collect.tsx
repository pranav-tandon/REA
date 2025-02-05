import { useState } from "react";
import { useRouter } from "next/router";

export default function CollectPage() {
  const router = useRouter();
  const [city, setCity] = useState("");
  const [stateVal, setStateVal] = useState("");
  const [price, setPrice] = useState("");
  const [actionType, setActionType] = useState("Buy");
  const [bedrooms, setBedrooms] = useState("");
  const [bathrooms, setBathrooms] = useState("");
  const [propertyType, setPropertyType] = useState("House");
  const [minSquareFeet, setMinSquareFeet] = useState("");
  const [maxSquareFeet, setMaxSquareFeet] = useState("");
  const [notes, setNotes] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/flowProxy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint: "/flow/collect-constraints",
          method: "POST",
          payload: {
            city,
            state: stateVal,
            price: parseFloat(price),
            actionType,
            bedrooms: parseInt(bedrooms),
            bathrooms: parseFloat(bathrooms),
            propertyType,
            minSquareFeet: minSquareFeet ? parseInt(minSquareFeet) : null,
            maxSquareFeet: maxSquareFeet ? parseInt(maxSquareFeet) : null,
            notes,
          },
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Submission failed");
      }
      if (data.profileId) {
        router.push(`/confirm?profileId=${data.profileId}`);
      } else {
        throw new Error("No profileId returned");
      }
    } catch (err: any) {
      console.error("Error collecting constraints:", err);
      setError(err.message || "Unknown error");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-md mx-auto bg-gray-800 rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold mb-6">REA- Home is where the heart is. Find Your Home</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              City <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              required
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="e.g. Austin"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              State <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={stateVal}
              onChange={(e) => setStateVal(e.target.value)}
              required
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="e.g. TX"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Action Type</label>
            <select
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              value={actionType}
              onChange={(e) => setActionType(e.target.value)}
            >
              <option value="Buy">Buy</option>
              <option value="Rent">Rent</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Price {actionType === 'Rent' ? '(Monthly)' : ''} <span className="text-red-400">*</span>
            </label>
            <input
              type="number"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              required
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder={actionType === 'Rent' ? 'e.g. 2000' : 'e.g. 500000'}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Property Type</label>
            <select
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              value={propertyType}
              onChange={(e) => setPropertyType(e.target.value)}
            >
              <option value="House">House</option>
              <option value="Apartment">Apartment</option>
              <option value="Condo">Condo</option>
              <option value="Townhouse">Townhouse</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Bedrooms <span className="text-red-400">*</span>
            </label>
            <select
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              value={bedrooms}
              onChange={(e) => setBedrooms(e.target.value)}
              required
            >
              <option value="">Select bedrooms</option>
              {[1, 2, 3, 4, 5, 6].map((num) => (
                <option key={num} value={num}>
                  {num} {num === 1 ? 'bedroom' : 'bedrooms'}
                </option>
              ))}
              <option value="7">7+ bedrooms</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Bathrooms <span className="text-red-400">*</span>
            </label>
            <select
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              value={bathrooms}
              onChange={(e) => setBathrooms(e.target.value)}
              required
            >
              <option value="">Select bathrooms</option>
              {[1, 1.5, 2, 2.5, 3, 3.5, 4].map((num) => (
                <option key={num} value={num}>
                  {num} {num === 1 ? 'bathroom' : 'bathrooms'}
                </option>
              ))}
              <option value="5">5+ bathrooms</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Square Footage (optional)</label>
            <div className="flex space-x-2">
              <input
                type="number"
                value={minSquareFeet}
                onChange={(e) => setMinSquareFeet(e.target.value)}
                className="w-1/2 p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="Min sq.ft"
              />
              <input
                type="number"
                value={maxSquareFeet}
                onChange={(e) => setMaxSquareFeet(e.target.value)}
                className="w-1/2 p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="Max sq.ft"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Notes (optional)</label>
            <textarea
              className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Any specific preferences?"
            />
          </div>

          {error && <p className="text-red-400">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50"
          >
            {loading ? "Submitting..." : "Next →"}
          </button>
        </form>
      </div>
    </div>
  );
}