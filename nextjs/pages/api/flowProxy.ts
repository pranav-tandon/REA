import type { NextApiRequest, NextApiResponse } from "next";

/**
 * A generic route that forwards requests to your FastAPI backend's /flow endpoints.
 * E.g. GET /flowProxy?endpoint=/flow/confirm&profileId=xxx
 *      POST /flowProxy => { endpoint, method, payload }
 */
export default async function flowProxy(req: NextApiRequest, res: NextApiResponse) {
  const backendBase = "http://localhost:8000";  // adjust if needed

  if (req.method === "GET") {
    // Example: /flowProxy?endpoint=/flow/confirm&profileId=xxx
    const endpoint = req.query.endpoint as string;
    const profileId = req.query.profileId as string;

    const url = `${backendBase}${endpoint}?profileId=${profileId}`;
    try {
      const backendRes = await fetch(url);
      const data = await backendRes.json();
      return res.status(backendRes.status).json(data);
    } catch (err) {
      console.error("Error in flowProxy GET:", err);
      return res.status(500).json({ error: "Failed to fetch from backend" });
    }
  } else if (req.method === "POST") {
    // Example body: { endpoint: '/flow/collect-constraints', method: 'POST', payload: {...} }
    const { endpoint, method, payload } = req.body;
    const url = `${backendBase}${endpoint}`;
    try {
      const backendRes = await fetch(url, {
        method: method || "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {})
      });
      const data = await backendRes.json();
      return res.status(backendRes.status).json(data);
    } catch (err) {
      console.error("Error in flowProxy POST:", err);
      return res.status(500).json({ error: "Failed to fetch from backend" });
    }
  } else {
    return res.status(405).json({ error: "Method not allowed" });
  }
} 