import {defineConfig} from 'vite';
export default defineConfig({server:{proxy:Object.fromEntries(['/physics','/training'].map(path=>[path,{target:'ws://127.0.0.1:8765',ws:true}]))}});
