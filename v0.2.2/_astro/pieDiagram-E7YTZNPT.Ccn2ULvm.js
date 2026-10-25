import"./src.B2_BndWA.js";import{t as e}from"./ordinal.BAxfPAgN.js";import{t}from"./arc.CVR29zhd.js";import{t as n}from"./pie.DG7NkDnX.js";import{n as r}from"./mermaid-parser.core.CbyKqkqW.js";import{n as i}from"./chunk-Y2CYZVJY.DsF7k-Jl.js";import{t as a}from"./chunk-X3CZISLH.CmQIif5t.js";import{H as o,K as s,U as c,a as l,c as u,f as d,v as f,w as p,x as m,y as h}from"./chunk-DU6HZSFF.BtlfzGrW.js";import{t as g}from"./chunk-JWPE2WC7.A3RzWITa.js";import{i as _,p as v}from"./chunk-75Z2AOVW.hoAdDA3W.js";import{f as y}from"./mermaid.core.D7KE8IvF.js";var b=d.pie,x={sections:new Map,showData:!1,config:b},S=x.sections,C=x.showData,w=structuredClone(b),T={getConfig:i(()=>structuredClone(w),`getConfig`),clear:i(()=>{S=new Map,C=x.showData,l()},`clear`),setDiagramTitle:s,getDiagramTitle:p,setAccTitle:c,getAccTitle:h,setAccDescription:o,getAccDescription:f,addSection:i(({label:e,value:t})=>{if(t<0)throw Error(`"${e}" has invalid value: ${t}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);S.has(e)||(S.set(e,t),a.debug(`added new section: ${e}, with value: ${t}`))},`addSection`),getSections:i(()=>S,`getSections`),setShowData:i(e=>{C=e},`setShowData`),getShowData:i(()=>C,`getShowData`)},E=i((e,t)=>{g(e,t),t.setShowData(e.showData),e.sections.map(t.addSection)},`populateDb`),D={parse:i(async e=>{let t=await r(`pie`,e);a.debug(t),E(t,T)},`parse`)},O=i(e=>`
  .pieCircle{
    stroke: ${e.pieStrokeColor};
    stroke-width : ${e.pieStrokeWidth};
    opacity : ${e.pieOpacity};
  }
  .pieCircle.highlighted{
    scale: 1.05;
    opacity: 1;
  }
  .pieCircle.highlightedOnHover:hover{
    transition-duration: 250ms;
    scale: 1.05;
    opacity: 1;
  }
  .pieOuterCircle{
    stroke: ${e.pieOuterStrokeColor};
    stroke-width: ${e.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${e.pieTitleTextSize};
    fill: ${e.pieTitleTextColor};
    font-family: ${e.fontFamily};
  }
  .slice {
    font-family: ${e.fontFamily};
    fill: ${e.pieSectionTextColor};
    font-size:${e.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${e.pieLegendTextColor};
    font-family: ${e.fontFamily};
    font-size: ${e.pieLegendTextSize};
  }
`,`getStyles`),k=i(e=>{let t=[...e.values()].reduce((e,t)=>e+t,0),r=[...e.entries()].map(([e,t])=>({label:e,value:t})).filter(e=>e.value/t*100>=1);return n().value(e=>e.value).sort(null)(r)},`createPieArcs`),A={parser:D,db:T,renderer:{draw:i((n,r,i,o)=>{a.debug(`rendering pie chart
`+n);let s=o.db,c=m(),l=_(s.getConfig(),c.pie),d=y(r),f=d.append(`g`);f.attr(`transform`,`translate(225,225)`);let{themeVariables:p}=c,[h]=v(p.pieOuterStrokeWidth);h??=2;let g=l.legendPosition,b=l.textPosition,x=l.donutHole>0&&l.donutHole<=.9?l.donutHole:0,S=t().innerRadius(x*185).outerRadius(185),C=t().innerRadius(185*b).outerRadius(185*b),w=f.append(`g`);w.append(`circle`).attr(`cx`,0).attr(`cy`,0).attr(`r`,185+h/2).attr(`class`,`pieOuterCircle`);let T=s.getSections(),E=k(T),D=[p.pie1,p.pie2,p.pie3,p.pie4,p.pie5,p.pie6,p.pie7,p.pie8,p.pie9,p.pie10,p.pie11,p.pie12],O=0;T.forEach(e=>{O+=e});let A=E.filter(e=>(e.data.value/O*100).toFixed(0)!==`0`),j=e(D).domain([...T.keys()]);w.selectAll(`mySlices`).data(A).enter().append(`path`).attr(`d`,S).attr(`fill`,e=>j(e.data.label)).attr(`class`,e=>{let t=`pieCircle`;return l.highlightSlice===`hover`?t+=` highlightedOnHover`:l.highlightSlice===e.data.label&&(t+=` highlighted`),t}),w.selectAll(`mySlices`).data(A).enter().append(`text`).text(e=>(e.data.value/O*100).toFixed(0)+`%`).attr(`transform`,e=>`translate(`+C.centroid(e)+`)`).style(`text-anchor`,`middle`).attr(`class`,`slice`);let M=f.append(`text`).text(s.getDiagramTitle()).attr(`x`,0).attr(`y`,-200).attr(`class`,`pieTitleText`),N=[...T.entries()].map(([e,t])=>({label:e,value:t})),P=f.selectAll(`.legend`).data(N).enter().append(`g`).attr(`class`,`legend`);P.append(`rect`).attr(`width`,18).attr(`height`,18).style(`fill`,e=>j(e.label)).style(`stroke`,e=>j(e.label)),P.append(`text`).attr(`x`,22).attr(`y`,14).text(e=>s.getShowData()?`${e.label} [${e.value}]`:e.label);let F=Math.max(...P.selectAll(`text`).nodes().map(e=>e?.getBoundingClientRect().width??0)),I=450,L=490,R=N.length*22;switch(g){case`center`:P.attr(`transform`,(e,t)=>{let n=22*N.length/2,r=-F/2-22,i=t*22-n;return`translate(`+r+`,`+i+`)`});break;case`top`:I+=R,P.attr(`transform`,(e,t)=>`translate(${-F/2-22}, ${t*22-185})`),w.attr(`transform`,()=>`translate(0, ${R+22})`);break;case`bottom`:I+=R,P.attr(`transform`,(e,t)=>{let n=-F/2-22,r=t*22- -207;return`translate(`+n+`,`+r+`)`});break;case`left`:L+=22+F,P.attr(`transform`,(e,t)=>{let n=22*N.length/2;return`translate(-207,`+(t*22-n)+`)`}),w.attr(`transform`,()=>`translate(${F+18+4}, 0)`);break;default:L+=22+F,P.attr(`transform`,(e,t)=>{let n=22*N.length/2;return`translate(216,`+(t*22-n)+`)`})}let z=M.node()?.getBoundingClientRect().width??0,B=225-z/2,V=225+z/2,H=Math.min(0,B),U=Math.max(L,V)-H;d.attr(`viewBox`,`${H} 0 ${U} ${I}`),u(d,I,U,l.useMaxWidth)},`draw`)},styles:O};export{A as diagram};