import {defineConfig} from 'vite';
export default defineConfig({server:{proxy:{'/physics':{target:'ws://127.0.0.1:8765',ws:true}}}});
