import { useRouter } from "next/router";
import { useEffect, useState } from "react";

export default function NeighborhoodsPage() {
  const router = useRouter();
  const { profileId } = router.query as { profileId?: string };

  const [recommended, setRecommended] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Fetch recommended neighborhoods from the confirm endpoint.
  useEffect(() => {
    if (profileId) {
      fetchConfirm(profileId);
    }
  }, [profileId]);

  const fetchConfirm = async (pid: string) => {
    try {
      const res = await fetch(`/api/flowProxy?endpoint=/flow/confirm&profileId=${pid}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to load neighborhoods");
      }
      setRecommended(data.recommended_neighborhoods || []);
    } catch (err: any) {
      console.error("Error fetching neighborhoods:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleNeighborhood = (nb: string) => {
    if (selected.includes(nb)) {
      setSelected(selected.filter((x) => x !== nb));
    } else {
      if (selected.length < 5) {
        setSelected([...selected, nb]);
      } else {
        alert("You can only select up to 5 neighborhoods");
      }
    }
  };

  const handleSubmit = async () => {
    if (!profileId) return;

    try {
      setLoading(true);
      setError("");

      // POST to /flow/select-neighborhoods with the formatted payload.
      const res = await fetch("/api/flowProxy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint: `/flow/select-neighborhoods`,
          method: "POST",
          payload: {
            profileId,
            neighborhoods: selected,
          },
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Error selecting neighborhoods");
      }

      router.push(`/final?profileId=${profileId}`);
    } catch (err: any) {
      console.error("Error selecting neighborhoods:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!profileId) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <p>No profileId found.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-xl mx-auto bg-gray-800 p-6 rounded shadow">
        <h1 className="text-2xl font-bold mb-4">Select Neighborhoods</h1>

        {error && <p className="text-red-400">{error}</p>}

        <div className="mb-4 space-y-2">
          {recommended.length === 0 && <p>No recommended neighborhoods found.</p>}
          {recommended.map((nb) => (
            <div key={nb} className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selected.includes(nb)}
                onChange={() => toggleNeighborhood(nb)}
              />
              <label>{nb}</label>
            </div>
          ))}
        </div>

        <button
          onClick={handleSubmit}
          className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-bold"
        >
          Finalize →
        </button>
      </div>
    </div>
  );
} 