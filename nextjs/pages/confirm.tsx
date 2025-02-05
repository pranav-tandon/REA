import { useState, useEffect } from "react";
import { useRouter } from "next/router";

export default function ConfirmPage() {
  const router = useRouter();
  const { profileId } = router.query as { profileId?: string };

  const [city, setCity] = useState("");
  const [stateVal, setStateVal] = useState("");
  const [cityStats, setCityStats] = useState<any>({});
  const [recommended, setRecommended] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (profileId) {
      fetchData(profileId);
    }
  }, [profileId]);

  const fetchData = async (pid: string) => {
    try {
      setLoading(true);
      setError("");
      const res = await fetch(`/api/flowProxy?endpoint=/flow/confirm&profileId=${pid}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to load confirmation data");
      }
      setCity(data.city || "");
      setStateVal(data.state || "");
      setCityStats(data.city_stats || {});
      setRecommended(data.recommended_neighborhoods || []);
    } catch (err: any) {
      console.error("Error fetching confirm data:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (!profileId) return;
    router.push(`/neighborhoods?profileId=${profileId}`);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <p>Loading confirmation data...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-xl mx-auto bg-gray-800 p-6 rounded shadow">
        <h1 className="text-2xl font-bold mb-4">Confirmation</h1>
        
        {error && <p className="text-red-500 mb-4">{error}</p>}

        <div className="mb-4">
          <p>
            <strong>City:</strong> {city}
          </p>
          <p>
            <strong>State:</strong> {stateVal}
          </p>
        </div>

        <div className="mb-4">
          <h2 className="text-lg font-semibold mb-2">City Stats</h2>
          <ul className="list-disc ml-5 space-y-1">
            {Object.entries(cityStats).map(([key, val]) => (
              <li key={key}>
                <strong>{key}:</strong> {val}
              </li>
            ))}
          </ul>
        </div>

        <div className="mb-4">
          <h2 className="text-lg font-semibold mb-2">Recommended Neighborhoods</h2>
          {recommended.length === 0 ? (
            <p>No recommended neighborhoods found.</p>
          ) : (
            <ul className="list-disc ml-5 space-y-1">
              {recommended.map((nb, idx) => (
                <li key={idx}>{nb}</li>
              ))}
            </ul>
          )}
        </div>

        <button
          onClick={handleNext}
          className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-bold"
        >
          Next →
        </button>
      </div>
    </div>
  );
}