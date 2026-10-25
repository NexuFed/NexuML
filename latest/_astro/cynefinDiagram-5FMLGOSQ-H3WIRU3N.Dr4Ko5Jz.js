import{n as e}from"./chunk-AMVFOWMQ.C_4fk-Dj.js";import{t}from"./chunk-YJFJOXZG.wtGER9Hz.js";import{E as n,J as r,Q as i,S as a,Y as o,b as s,k as c,l,m as u,o as d,x as f}from"./chunk-W7FHEGFS.CvoMBbdl.js";import{t as p}from"./chunk-VII2H2IX.9E14KE34.js";import{M as m}from"./chunk-KOCW2XDZ.BqoF1hYF.js";import"./chunk-WGJ4HU3W.vZSIfS91.js";import{i as h}from"./chunk-S2UQUSRU.As3sCcV3.js";import{p as g}from"./render-O7CIS3YK.DfHZqklC.js";import{r as _}from"./mermaid-layout-elk.core.B6NU23ls.js";var v=e(()=>({domains:new Map,transitions:[]}),`createDefaultData`),y=v(),b={getDomains:e(()=>y.domains,`getDomains`),getTransitions:e(()=>y.transitions,`getTransitions`),setDomains:e(e=>{if(e)for(let t of e){let e=t.domain,n=(t.items??[]).map(e=>({label:e.label}));y.domains.set(e,{name:e,items:n})}},`setDomains`),setTransitions:e(e=>{e&&(y.transitions=e.filter(e=>e.from!==e.to||(t.warn(`Cynefin: self-loop transition on domain "${e.from}" is not meaningful and will be skipped.`),!1)).map(e=>({from:e.from,to:e.to,label:e.label||void 0})))},`setTransitions`),getConfig:e(()=>h({...u.cynefin,...a().cynefin}),`getConfig`),clear:e(()=>{d(),y=v()},`clear`),setAccTitle:o,getAccTitle:f,setDiagramTitle:i,getDiagramTitle:n,getAccDescription:s,setAccDescription:r},x=e(e=>{p(e,b),b.setDomains(e.domains),b.setTransitions(e.transitions)},`populate`),S={parse:e(async e=>{let n=await m(`cynefin`,e);t.debug(n),x(n)},`parse`)};function C(e){let t=e+1831565813|0;return t=Math.imul(t^t>>>15,t|1),t^=t+Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}_(C,`seededRandom`),e(C,`seededRandom`);function w(e){let t=0;for(let n=0;n<e.length;n++){let r=e.charCodeAt(n);t=(t<<5)-t+r,t|=0}return t}_(w,`hashString`),e(w,`hashString`);function T(e,t){return typeof e==`number`&&Number.isFinite(e)&&e!==0?e:w(t)}_(T,`resolveSeed`),e(T,`resolveSeed`);function E(e,t,n,r){let i=e/2,a=r??e*.015,o=t/7,s=[];for(let e=0;e<=7;e++){let t=C(n+e*17)*a*2-a;s.push({x:i+t,y:e*o})}let c=`M${s[0].x},${s[0].y}`;for(let e=0;e<s.length-1;e++){let t=s[e],r=s[e+1],i=(t.y+r.y)/2,o=e%2==0?1:-1,l=a*1.5*o*C(n+e*31+7),u=t.x+l,d=i,f=r.x-l;c+=` C${u},${d} ${f},${i} ${r.x},${r.y}`}return c}_(E,`generateFoldPath`),e(E,`generateFoldPath`);function D(e,t,n,r){let i=t/2,a=r??t*.015,o=e/7,s=[];for(let e=0;e<=7;e++){let t=C(n+e*23)*a*2-a;s.push({x:e*o,y:i+t})}let c=`M${s[0].x},${s[0].y}`;for(let e=0;e<s.length-1;e++){let t=s[e],r=s[e+1],i=(t.x+r.x)/2,o=e%2==0?1:-1,l=a*1.5*o*C(n+e*37+11),u=i,d=t.y+l,f=i,p=r.y-l;c+=` C${u},${d} ${f},${p} ${r.x},${r.y}`}return c}_(D,`generateHorizontalBoundary`),e(D,`generateHorizontalBoundary`);function O(e,t){let n=e/2,r=t*.5,i=t,a=e*.03;return[`M${n},${r}`,`C${n+a},${r+(i-r)*.2}`,`${n-a*1.5},${r+(i-r)*.55}`,`${n+a*.5},${r+(i-r)*.75}`,`C${n-a},${r+(i-r)*.85}`,`${n+a*.3},${r+(i-r)*.95}`,`${n},${i}`].join(` `)}_(O,`generateCliffPath`),e(O,`generateCliffPath`);function k(e,t,n,r){return[`M${e-n},${t}`,`A${n},${r} 0 1,1 ${e+n},${t}`,`A${n},${r} 0 1,1 ${e-n},${t}`,`Z`].join(` `)}_(k,`generateConfusionPath`),e(k,`generateConfusionPath`);var A={complex:{model:`Probe → Sense → Respond`,practice:`Emergent Practices`},complicated:{model:`Sense → Analyse → Respond`,practice:`Good Practices`},clear:{model:`Sense → Categorise → Respond`,practice:`Best Practices`},chaotic:{model:`Act → Sense → Respond`,practice:`Novel Practices`},confusion:{model:``,practice:`Disorder`}},j=e((e,t)=>{let n=e/2,r=t/2;return{complex:{cx:n/2,cy:r/2,x:0,y:0,w:n,h:r},complicated:{cx:n+n/2,cy:r/2,x:n,y:0,w:n,h:r},chaotic:{cx:n/2,cy:r+r/2,x:0,y:r,w:n,h:r},clear:{cx:n+n/2,cy:r+r/2,x:n,y:r,w:n,h:r},confusion:{cx:n,cy:r,x:n*.7,y:r*.7,w:n*.6,h:r*.6}}},`getDomainLayouts`),M=e(()=>{let e=c(),t=a();return h(e,t.themeVariables).cynefin},`getCynefinDomainColors`),N=3,P={draw:e((e,n,r,i)=>{let a=i.db,o=a.getDomains(),s=a.getTransitions(),c=a.getDiagramTitle(),u=a.getAccTitle(),d=a.getAccDescription(),f=a.getConfig(),p=M();t.debug(`Rendering Cynefin diagram`);let m=f.width,h=f.height,_=f.padding,v=f.showDomainDescriptions,y=f.boundaryAmplitude,b=m+_*2,x=h+_*2,S={complex:p.complexBg,complicated:p.complicatedBg,clear:p.clearBg,chaotic:p.chaoticBg,confusion:p.confusionBg},C=g(n);l(C,x,b,f.useMaxWidth??!0),C.attr(`viewBox`,`0 0 ${b} ${x}`),u&&C.append(`title`).text(u),d&&C.append(`desc`).text(d);let w=C.append(`g`).attr(`transform`,`translate(${_}, ${_})`),P=j(m,h),F=T(f.seed,n),I=w.append(`g`).attr(`class`,`cynefin-backgrounds`),L=[`complex`,`complicated`,`chaotic`,`clear`];for(let e of L){let t=P[e];I.append(`rect`).attr(`class`,`cynefinDomain`).attr(`x`,t.x).attr(`y`,t.y).attr(`width`,t.w).attr(`height`,t.h).attr(`fill`,S[e]).attr(`fill-opacity`,.4).attr(`stroke`,`none`)}let R=w.append(`g`).attr(`class`,`cynefin-boundaries`);R.append(`path`).attr(`class`,`cynefinBoundary`).attr(`d`,E(m,h,F,y)).attr(`fill`,`none`),R.append(`path`).attr(`class`,`cynefinBoundary`).attr(`d`,D(m,h,F+100,y)).attr(`fill`,`none`),R.append(`path`).attr(`class`,`cynefinCliff`).attr(`d`,O(m,h)).attr(`fill`,`none`);let z=m*.15,B=h*.15;w.append(`path`).attr(`class`,`cynefinConfusion`).attr(`d`,k(m/2,h/2,z,B)).attr(`fill`,S.confusion).attr(`fill-opacity`,.5);let V=w.append(`g`).attr(`class`,`cynefin-labels`);for(let e of L){let t=P[e];V.append(`text`).attr(`class`,`cynefinDomainLabel`).attr(`x`,t.cx).attr(`y`,v?t.cy-30:t.cy).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`middle`).text(e.charAt(0).toUpperCase()+e.slice(1))}if(V.append(`text`).attr(`class`,`cynefinDomainLabel`).attr(`x`,m/2).attr(`y`,v?h/2-10:h/2).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`middle`).text(`Confusion`),v){let e=w.append(`g`).attr(`class`,`cynefin-subtitles`);for(let t of L){let n=P[t],r=A[t];e.append(`text`).attr(`class`,`cynefinSubtitle`).attr(`x`,n.cx).attr(`y`,n.cy-10).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`middle`).text(r.model),e.append(`text`).attr(`class`,`cynefinSubtitle`).attr(`x`,n.cx).attr(`y`,n.cy+5).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`middle`).text(r.practice)}e.append(`text`).attr(`class`,`cynefinSubtitle`).attr(`x`,m/2).attr(`y`,h/2+8).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`middle`).text(A.confusion.practice)}let H=w.append(`g`).attr(`class`,`cynefin-items`);for(let e of[`complex`,`complicated`,`chaotic`,`clear`,`confusion`]){let t=o.get(e);if(!t||t.items.length===0)continue;let n=P[e],r=e===`confusion`,i=t.items,a=0;r&&t.items.length>N&&(a=t.items.length-N,i=t.items.slice(0,N));let s;if(r){let e=v?22:14;s=n.cy+e}else s=n.cy+(v?25:15);if([...i].forEach((t,r)=>{let i=s+r*30,a=H.append(`g`),o=a.append(`text`).attr(`class`,`cynefinItemText`).attr(`x`,0).attr(`y`,13).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`central`).text(t.label),c=t.label.length*7,l=o.node();if(l&&typeof l.getBBox==`function`){let e=l.getBBox();e.width>0&&(c=e.width)}let u=c+20,d=n.cx-u/2;a.attr(`transform`,`translate(${d}, ${i})`),a.insert(`rect`,`text`).attr(`class`,`cynefinItem`).attr(`x`,0).attr(`y`,0).attr(`width`,u).attr(`height`,26).attr(`rx`,4).attr(`ry`,4).attr(`fill`,S[e]).attr(`fill-opacity`,.95),o.attr(`x`,u/2).attr(`y`,13)}),a>0){let t=s+i.length*30,r=`+${a} more`,o=H.append(`g`),c=o.append(`text`).attr(`class`,`cynefinItemText`).attr(`x`,0).attr(`y`,13).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`central`).text(r),l=r.length*7,u=c.node();if(u&&typeof u.getBBox==`function`){let e=u.getBBox();e.width>0&&(l=e.width)}let d=l+20,f=n.cx-d/2;o.attr(`transform`,`translate(${f}, ${t})`),o.insert(`rect`,`text`).attr(`class`,`cynefinItemOverflow`).attr(`x`,0).attr(`y`,0).attr(`width`,d).attr(`height`,26).attr(`rx`,4).attr(`ry`,4).attr(`fill`,S[e]).attr(`fill-opacity`,.6),c.attr(`x`,d/2).attr(`y`,13)}}if(s.length>0){let e=C.select(`defs`).empty()?C.append(`defs`):C.select(`defs`),r=`cynefin-arrow-${n}`;e.append(`marker`).attr(`id`,r).attr(`viewBox`,`0 0 10 10`).attr(`refX`,9).attr(`refY`,5).attr(`markerWidth`,6).attr(`markerHeight`,6).attr(`orient`,`auto-start-reverse`).append(`path`).attr(`d`,`M 0 0 L 10 5 L 0 10 z`).attr(`class`,`cynefinArrowHead`);let i=w.append(`g`).attr(`class`,`cynefin-arrows`);s.forEach(e=>{let n=P[e.from],a=P[e.to];if(!n||!a)return;if(e.from===e.to){t.warn(`Cynefin renderer: skipping self-loop on domain "${e.from}"`);return}let o=n.cx,s=n.cy,c=a.cx,l=a.cy,u=(o+c)/2,d=(s+l)/2,f=c-o,p=l-s,m=Math.sqrt(f*f+p*p),h=m*.15,g=-p/m,_=f/m,v=u+g*h,y=d+_*h;i.append(`path`).attr(`class`,`cynefinArrowLine`).attr(`d`,`M${o},${s} Q${v},${y} ${c},${l}`).attr(`fill`,`none`).attr(`marker-end`,`url(#${r})`),e.label&&i.append(`text`).attr(`class`,`cynefinArrowLabel`).attr(`x`,v).attr(`y`,y-6).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`auto`).text(e.label)})}c&&w.append(`text`).attr(`class`,`cynefinTitle`).attr(`x`,m/2).attr(`y`,-_/2).attr(`text-anchor`,`middle`).attr(`dominant-baseline`,`middle`).text(c)},`draw`)},F=e(()=>{let e=c(),t=a();return h(e,t.themeVariables).cynefin},`getCynefinTheme`),I={parser:S,db:b,renderer:P,styles:e(()=>{let e=F();return`
	.cynefinDomain {
		stroke: none;
	}
	.cynefinDomainLabel {
		font-size: ${e.domainFontSize}px;
		font-weight: bold;
		fill: ${e.labelColor};
	}
	.cynefinSubtitle {
		font-size: ${e.itemFontSize-1}px;
		fill: ${e.textColor};
		font-style: italic;
	}
	.cynefinItem {
		fill-opacity: 0.95;
		stroke: ${e.boundaryColor};
		stroke-width: 1;
	}
	.cynefinItemText {
		font-size: ${e.itemFontSize}px;
		fill: ${e.textColor};
	}
	.cynefinItemOverflow {
		fill-opacity: 0.6;
		stroke: ${e.boundaryColor};
		stroke-width: 1;
		stroke-dasharray: 3 2;
	}
	.cynefinBoundary {
		stroke: ${e.boundaryColor};
		stroke-width: ${e.boundaryWidth};
		stroke-dasharray: 6 3;
	}
	.cynefinCliff {
		stroke: ${e.cliffColor};
		stroke-width: ${e.cliffWidth};
	}
	.cynefinConfusion {
		stroke: ${e.boundaryColor};
		stroke-width: 1.5;
		stroke-dasharray: 4 2;
	}
	.cynefinArrowLine {
		stroke: ${e.arrowColor};
		stroke-width: ${e.arrowWidth};
		fill: none;
	}
	.cynefinArrowHead {
		fill: ${e.arrowColor};
		stroke: none;
	}
	.cynefinArrowLabel {
		font-size: ${e.itemFontSize-1}px;
		fill: ${e.textColor};
	}
	.cynefinTitle {
		font-size: ${e.domainFontSize+2}px;
		font-weight: bold;
		fill: ${e.labelColor};
	}
	`},`styles`)};export{I as diagram};