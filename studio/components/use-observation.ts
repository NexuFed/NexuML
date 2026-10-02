"use client";
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { ConnectionInfo, Event, Operation } from "../model/types";

export function useObservation(connection:ConnectionInfo, identity:string) {
  const client = useQueryClient();
  const [observed, setObserved] = useState<{id:string;events:Event[];state:string;truncated:boolean;gap:boolean}>({id:"",events:[],state:"Detached",truncated:false,gap:false});
  useEffect(()=>{
    if(!identity) return;
    let closed=false, cursor=0, socket:WebSocket, retry:ReturnType<typeof setTimeout>;
    const update = (patch:Partial<typeof observed>) => setObserved(previous=>({...previous,...patch,id:identity}));
    update({events:[],state:"Connecting",truncated:false,gap:false});
    const attach=()=>{
      const url=new URL(`${connection.api}/api/v1/operations/${identity}/events`);
      url.protocol="ws:";url.searchParams.set("after",String(cursor));
      socket=new WebSocket(url);
      socket.onopen=()=>{socket.send(JSON.stringify({token:connection.token}));update({state:"Connected"});};
      socket.onmessage=message=>{
        const event:Event=JSON.parse(message.data);
        if(event.kind==="snapshot") {
          if(event.partial)void client.invalidateQueries({queryKey:["operation",identity]});
          else client.setQueryData(["operation",identity],event.payload as unknown as Operation);
        }
        if(event.sequence && event.sequence<=cursor) return;
        if(event.sequence) cursor=event.sequence;
        if(event.kind==="terminal") void client.invalidateQueries({queryKey:["operation",identity]});
        if(event.kind==="replay_gap") update({gap:true});
        setObserved(previous=>({id:identity,state:previous.id===identity ? previous.state : "Connected",
          events:[...(previous.id===identity?previous.events:[]),event].slice(-2000),
          truncated:previous.truncated || previous.events.length>=2000,gap:previous.gap}));
      };
      socket.onclose=event=>{
        if(closed) return;
        update({state:event.code===1000 ? "Complete" : "Disconnected — retrying"});
        if(event.code!==1000 && event.code!==1008) retry=setTimeout(attach,1500);
        if(event.code===1008) update({state:"Access/cursor rejected — reload to reconnect"});
      };
      socket.onerror=()=>update({state:"Connection interrupted"});
    };
    attach();
    return ()=>{closed=true;clearTimeout(retry);socket?.close();};
  },[connection,identity,client]);
  return observed.id===identity ? observed : {events:[],state:"Detached",truncated:false,gap:false};
}
